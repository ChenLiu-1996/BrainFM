from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from einops import rearrange


class SpatialGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_channels, hidden_channels))

    def forward(self, data_list):
        '''
        data_list: List of PyG Data objects, one per frame in a sequence
        Returns: Tensor of shape [N, V, hidden_dim]
        '''
        encoded = []
        for data in data_list:
            x, edge_index = data.x, data.edge_index
            for layer in self.layers:
                x = layer(x, edge_index)
                x = torch.relu(x)
            encoded.append(x)
        return torch.stack(encoded, dim=0)  # [N, V, d]

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        '''
        x: [B, N, V, d] → flatten to [B, N*V, d] or reshape to [B, N, V*d]
        Here we treat each timepoint as a token by flattening the spatial dim
        '''
        B, N, V, d = x.shape
        x = x.view(B, N, V * d)  # [B, N, V*d]
        x = self.encoder(x)     # [B, N, V*d]
        return x

class VideoDecoder(nn.Module):
    def __init__(self, input_dim, out_res=(256, 256)):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_res[0] * out_res[1] * 3)
        )
        self.out_res = out_res

    def forward(self, x):
        '''
        x: [B, 6N, d]
        '''
        B, T, d = x.shape
        x = self.linear(x)
        x = x.view(B, T, self.out_res[0], self.out_res[1], 3)
        return x

class SpatialTemporalGraphTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 nhead,
                 num_transformer_layers,
                 input_frames: int,
                 temporal_upsampling: int = 6,
                 out_res: Tuple[int] = (256, 256)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_encoder = GATv2Conv(in_channels, hidden_dim, edge_dim=2)  # edge_attr=[E, 2]

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Learnable temporal query tokens for upsampling
        self.temporal_queries = nn.Parameter(torch.randn(1, input_frames * temporal_upsampling, hidden_dim))

        # Decoder: maps each temporal embedding to a video frame
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_res[0] * out_res[1] * 3)
        )
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
            video: [B, 6N, H, W, 3] predicted frames
        '''
        x = self.spatial_encoder(x=data.x,
                                 edge_index=data.edge_index,
                                 edge_attr=data.edge_attr)                 # [B * T * V, hidden_dim]

        # Get batch size, time steps, and reshape
        B = data.ptr.numel() - 1
        T = int(data.time.max().item() + 1)
        V = x.size(0) // (B * T)

        # Flatten spatial dimension
        x = x.view(B, T * V, self.hidden_dim)                              # [B, T * V, hidden_dim]
        x = self.temporal_encoder(x)                                       # [B, T * V, d]

        # Expand temporally to 6N using query tokens
        upsample_factor = 6
        T_out = T * upsample_factor
        queries = self.temporal_queries.expand(B, T_out, -1)               # [B, temporal_upsampling * T, d]

        # Cross-attend queries to encoded sequence.
        mean_context = x.mean(dim=1, keepdim=True)                         # [B, 1, d]
        x_expanded = self.temporal_encoder(queries + 0.01 * mean_context)  # [B, temporal_upsampling * T, d]

        # Decode each temporal embedding to a video frame
        video = self.decoder(x_expanded)                                   # [B, 6N, H * W * 3]
        video = video.view(B, T_out, self.out_res[0], self.out_res[1], 3)  # [B, temporal_upsampling * T, H, W, 3]
        video = rearrange(video, 'b t h w d -> b h w d t')                 # [B, H, W, 3, temporal_upsampling * T]
        return video