from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from einops import rearrange
from diffusers import StableVideoDiffusionPipeline


class SpatialGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, edge_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_channels, hidden_dim, edge_dim=edge_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim))

    def forward(self, x, edge_index, edge_attr):
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if layer_idx < len(self.layers) - 1:
                x = torch.nn.functional.silu(x)
        return x


class LatentTokenDecoder(nn.Module):
    def __init__(self, token_dim: int = 1024, output_frames: int = 6, out_channels: int = 4, size_hw: int = 32):
        super().__init__()
        self.num_tokens = output_frames * out_channels
        self.out_channels = out_channels
        self.size_hw = size_hw
        self.output_frames = output_frames
        self.token_proj = nn.Linear(token_dim, size_hw * size_hw)

    def forward(self, tokens):
        B, _, _ = tokens.shape                                                            # Expecting [B, num_tokens, token_dim]
        x = self.token_proj(tokens)                                                       # [B, num_tokens, h * w]
        x = x.view(B, self.output_frames, self.out_channels, self.size_hw, self.size_hw)  # [B, output_frames, c, h, w]
        return x


class VideoDecoderSVD(nn.Module):
    '''
    The video decoder uses Stable Video Diffusion.
    https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py
    '''
    def __init__(self,
                 batch_size: int,
                 input_frames: int,
                 temporal_upsampling: int = 6,
                 fps: int = 3,
                 out_res: Tuple[int] = (256, 256),
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.output_frames = input_frames * temporal_upsampling
        self.out_res = out_res
        self.device = device

        svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16).to(device)

        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(1)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        self.fps = fps - 1

        self.added_time_ids = svd._get_add_time_ids(
            self.fps,
            motion_bucket_id=127,
            noise_aug_strength=0,
            dtype=torch.float32,
            batch_size=batch_size,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False).to(device)

        self.initial_latents = svd.prepare_latents(
            batch_size, num_frames=self.output_frames,
            num_channels_latents=svd.unet.config.in_channels,
            height=out_res[0], width=out_res[1],
            dtype=torch.float32, device=device, generator=torch_generator, latents=None)

        self.unet = svd.unet
        self.scheduler = svd.scheduler
        self.decode_latents = svd.decode_latents

        self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, image_latents, image_embeddings):
        '''
        image_latents: [B, output_frames, c, h, w]
        image_embeddings: [B, 1, 1024]
        For Stable Video diffusion, c = 4, h = w = 32.
        '''

        # Setup scheduler. Typical inference: 25 steps
        self.scheduler.set_timesteps(25, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = self.initial_latents
        for t in timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # Predict the noise residual.
            noise_pred = self.unet(
                latent_model_input.to(self.device).half(),
                t.to(self.device).half(),
                encoder_hidden_states=image_embeddings.to(self.device).half(),
                added_time_ids=self.added_time_ids.to(self.device).half(),
                return_dict=False)[0]

            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # [B, C, output_frames, H, W]
        video_pred = self.decode_latents(latents.half(), num_frames=self.output_frames)

        return video_pred


class SpatialTemporalGraphTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 input_frames: int,
                 temporal_upsampling: int = 6,
                 out_res: Tuple[int] = (256, 256),
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_upsampling = temporal_upsampling
        self.output_frames = input_frames * temporal_upsampling
        self.token_dim = 1024

        self.spatial_encoder = SpatialGraphEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            edge_dim=2)

        self.to_token_dim = nn.Linear(hidden_dim, self.token_dim)
        self.latent_tokens = nn.Parameter(torch.randn(1, self.output_frames * 4, self.token_dim))
        self.token_attn = nn.MultiheadAttention(self.token_dim, num_heads=4, batch_first=True)
        self.to_svd_embedding = LatentTokenDecoder(self.token_dim, output_frames=self.output_frames, out_channels=4, size_hw=32)

        self.decoder = VideoDecoderSVD(
            batch_size=1,
            out_res=out_res,
            input_frames=input_frames,
            temporal_upsampling=temporal_upsampling,
            device=device)

        self.out_res = out_res

    def forward(self, data):
        '''
        Args:
            data: PyG DataBatch with:
                - x: [B * T * V, in_channels]
                - edge_index: [2, E]
                - edge_attr: [E, 2]
                - time: [B * T * V]
                - batch: [B * T * V]

        Returns:
            video: [B, T, H, W, 3] predicted frames
                   Note that the number of frames T may be upsampled.
        '''
        x = self.spatial_encoder(x=data.x,
                                 edge_index=data.edge_index,
                                 edge_attr=data.edge_attr)                 # [B * T * V, d]

        # Get batch size, time steps, and reshape
        B = data.ptr.numel() - 1                                           # B: batch size
        T = int(data.time.max().item() + 1)                                # T: number of input frames
        V = x.size(0) // (B * T)                                           # V: number of voxels/vertices

        x = rearrange(x, '(b t v) d -> b (t v) d', b=B, t=T, v=V)          # [B, V*T, d]
        x = self.to_token_dim(x)                                           # [B, V*T, token_dim]
        x, _ = self.token_attn(self.latent_tokens, x, x)                   # [B, num_tokens, token_dim]
        image_embeddings = x[:, :1, :]                                     # [B, 1, token_dim]
        image_latents = self.to_svd_embedding(x)                           # [B, output_frames, out_channels, height, width]

        video = self.decoder(image_latents, image_embeddings)              # [B, C, output_frames, H, W]
        video = rearrange(video, 'b c t h w -> b h w c t')                 # [B, H, W, 3, output_frames]
        return video


if __name__ == '__main__':
    import os
    import sys
    import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
    sys.path.insert(0, import_dir + '/src/dataset/')
    from dynamic_natural_vision import DynamicNaturalVisionDataset
    from torch_geometric.loader import DataLoader

    test_set = DynamicNaturalVisionDataset(
        subject_idx=None,
        fMRI_window_frames=1,
        graph_knn_k=5,
        mode='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = SpatialTemporalGraphTransformer(
        in_channels=1,
        hidden_dim=16,
        input_frames=1,
        temporal_upsampling=6,
    )

    data_item = next(iter(test_loader))
    model(data_item[0])