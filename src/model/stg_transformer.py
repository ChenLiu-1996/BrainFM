from __future__ import annotations

from typing import Protocol, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from einops import rearrange

LATENT_SCALING = 0.18215  # Stable Diffusion constant


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(cond_dim, dim, bias=False)
        self.v = nn.Linear(cond_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor, cond: torch.Tensor):  # x: [B,C,H,W,T]
        B, C, H, W, T = x.shape
        q = self.q(x.permute(0, 4, 2, 3, 1).reshape(B * T * H * W, C))
        k = self.k(cond).unsqueeze(1)
        v = self.v(cond).unsqueeze(1)
        attn = (q.unsqueeze(-2) * self.scale) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(-2)
        out = self.proj(out).view(B, T, H, W, C).permute(0, 4, 2, 3, 1)
        return x + out

class TemporalUNet3D(nn.Module):
    def __init__(self, clip_dim: int, base_channels: int):
        super().__init__()
        self.down = nn.Conv3d(4, base_channels, 3, padding=1)
        self.attn = CrossAttentionBlock(base_channels, clip_dim)
        self.up = nn.ConvTranspose3d(base_channels, 4, 3, padding=1)

    def forward(self, z: torch.Tensor, clip_tokens: torch.Tensor):
        h = F.relu(self.down(z))
        h = self.attn(h, clip_tokens)
        return torch.tanh(self.up(h))

    def generate(self, clip_tokens: torch.Tensor, *, num_frames: int, steps: int = 20):
        B = clip_tokens.size(0)
        z = torch.randn(B, 4, 32, 32, num_frames, device=clip_tokens.device)
        for _ in range(steps):
            z = z - 0.05 * self.forward(z, clip_tokens)
        return z.permute(0, 4, 1, 2, 3).contiguous()  # (B,T,4,32,32)

class GraphSpatiotemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, g_hid: int, tok_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, g_hid)
        self.convs = nn.ModuleList([
            GATv2Conv(g_hid, g_hid, heads=4, concat=False, edge_dim=2) for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(g_hid) for _ in range(3)])
        self.node2tok = nn.Linear(g_hid * 2, tok_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(tok_dim, 8, 4 * tok_dim, batch_first=True), 2
        )

    def forward(self, graphs: List[Data]):
        if isinstance(graphs, Data):
            graphs = graphs.to_data_list()
        tokens = []
        for g in graphs:
            x = self.proj(g.x)
            for conv, ln in zip(self.convs, self.norms):
                x = ln(F.relu(conv(x, g.edge_index, g.edge_attr)) + x)
            pooled = torch.cat([
                global_mean_pool(x, g.batch), global_max_pool(x, g.batch)
            ], 1)
            tokens.append(self.node2tok(pooled))
        token_appearance = torch.stack(tokens, 1)  # [B,T,512]
        if token_appearance.size(1) > 1:
            token_motion = token_appearance[:, 1:] - token_appearance[:, :-1]
        else:
            token_motion = token_appearance.new_zeros(token_appearance.size(0), 0, token_appearance.size(-1) // 2)
        token_appearance = self.transformer(token_appearance)
        return token_appearance, token_motion[..., :256]

class BrainToCLIPTokens(nn.Module):
    def __init__(self, tok_dim: int, clip_dim: int = 768, max_tokens: int = 77):
        super().__init__()
        self.proj_appearance = nn.Linear(tok_dim, clip_dim)
        self.proj_motion = nn.Linear(tok_dim // 2, clip_dim)
        self.null = nn.Parameter(torch.randn(1, max_tokens, clip_dim) * 0.02)

    def forward(self, token_appearance: torch.Tensor, token_motion: torch.Tensor):
        brain = self.proj_appearance(token_appearance)
        if token_motion.nelement():
            brain = torch.cat([brain, self.proj_motion(token_motion)], 1)
        B, N, _ = brain.shape
        pad = self.null[:, : 77 - N].expand(B, -1, -1)
        return torch.cat([brain, pad], 1)

class SpatialTemporalGraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 96,
        input_frames: int = 1,
        temporal_upsampling: int = 6,
        device: Union[str, torch.device] = "cuda",
    ):
        '''
        End-to-end fMRI-to-video pipeline.

        Args:
            in_channels: voxel feature channels (usually 1).
            hidden_dim: graph hidden size (g_hid).
            input_frames: number of fMRI frames per sample.
            temporal_upsampling: how many video frames to generate per fMRI frame.
            device: torch device.
        '''
        super().__init__()
        self.input_frames = input_frames
        self.temporal_upsampling = temporal_upsampling
        self.clip_dim = 768

        # sub-modules
        self.encoder = GraphSpatiotemporalEncoder(in_channels, hidden_dim, tok_dim=512)
        self.projector = BrainToCLIPTokens(tok_dim=512, clip_dim=self.clip_dim)
        self.latent_gen = TemporalUNet3D(clip_dim=self.clip_dim, base_channels=hidden_dim * 8)
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32
        ).to(device).eval().requires_grad_(False)

        self.to(device)

    def forward(self, graphs: List[Data]):
        token_appearance, token_motion = self.encoder(graphs)
        tokens = self.projector(token_appearance, token_motion)
        video_frames = self.temporal_upsampling * self.input_frames
        latents = self.latent_gen.generate(tokens, num_frames=video_frames)
        latents = latents / LATENT_SCALING
        imgs = self.vae.decode(rearrange(latents, 'b t c h w -> (b t) c h w')).sample
        return rearrange(imgs, '(b t) c h w -> b h w c t', b=latents.shape[0])


if __name__ == "__main__":
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
        hidden_dim=96,
        input_frames=1,
        temporal_upsampling=6,
        device='cpu',
    )

    data_item = next(iter(test_loader))
    video = model(data_item[0])
    print('generated', video.shape)
