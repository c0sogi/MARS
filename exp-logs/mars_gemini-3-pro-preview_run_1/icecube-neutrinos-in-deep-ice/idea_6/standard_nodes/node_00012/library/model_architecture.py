import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    knn_graph,
    global_mean_pool,
    GlobalAttention,
)
from library.config import Config


class FourierEmbedding(nn.Module):
    """
    Encodes spatiotemporal coordinates (x, y, z, t) using sinusoidal functions
    of varying frequencies to capture high-frequency geometric details.
    """

    def __init__(self, num_freqs=6, input_dim=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        self.freqs = torch.pow(2, torch.arange(num_freqs)).float()

    def forward(self, x):
        # x shape: (N, input_dim)
        # Create grid of frequencies
        freqs = self.freqs.to(x.device)  # (L,)

        # Calculate arguments: x * pi * freq
        # unsqueeze to broadcast: (N, input_dim, 1) * (1, 1, L)
        args = x.unsqueeze(-1) * np.pi * freqs.view(1, 1, -1)  # (N, input_dim, L)

        # Apply sin and cos
        sin_feat = torch.sin(args)
        cos_feat = torch.cos(args)

        # Concatenate: (N, input_dim, L, 2)
        emb = torch.stack([sin_feat, cos_feat], dim=-1)

        # Flatten last dimensions: (N, input_dim * L * 2)
        emb = emb.view(x.size(0), -1)
        return emb


import numpy as np  # Needed for np.pi inside the class


class GlobalExchange(nn.Module):
    """
    Aggregates global event context and broadcasts it back to local nodes.
    Virtual Global Node mechanism.
    """

    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, batch):
        # x: (Total_Nodes, C)
        # batch: (Total_Nodes,)

        # 1. Global Pooling (Aggregate all pulses in event)
        global_feat = global_mean_pool(x, batch)  # (Batch_Size, C)

        # 2. Transform Global Context
        global_feat = self.mlp(global_feat)

        # 3. Broadcast and Add (Residual)
        # global_feat[batch] expands (B, C) -> (Total_Nodes, C)
        out = x + global_feat[batch]

        return self.norm(out)


class DynGTBlock(nn.Module):
    """
    Dynamic Graph Transformer Block.
    1. Dynamic k-NN graph construction in latent space.
    2. Graph Transformer Convolution.
    3. Global Exchange.
    4. Feed Forward Network.
    """

    def __init__(self, channels, heads, k_knn, dropout=0.1):
        super().__init__()
        self.k_knn = k_knn

        # Transformer Conv
        # Output dim will be heads * (channels // heads) = channels
        self.conv = TransformerConv(
            in_channels=channels,
            out_channels=channels // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            beta=True,  # Use bias
        )

        self.norm1 = nn.LayerNorm(channels)

        # Global Exchange
        self.global_exchange = GlobalExchange(channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x, batch):
        # 1. Dynamic Graph Construction
        # Compute k-NN graph based on current features
        edge_index = knn_graph(x, k=self.k_knn, batch=batch, loop=False)

        # 2. Graph Transformer Conv
        h = self.conv(x, edge_index)
        x = self.norm1(x + h)  # Residual

        # 3. Global Exchange
        x = self.global_exchange(x, batch)

        # 4. FFN
        h = self.ffn(x)
        x = self.norm2(x + h)  # Residual

        return x


class DynGTNet(nn.Module):
    """
    Main Dynamic Graph Transformer Network.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.in_channels = Config.IN_CHANNELS
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.num_heads = Config.NUM_HEADS
        self.num_layers = Config.NUM_LAYERS
        self.k_knn = Config.K_KNN
        self.dropout = Config.DROPOUT

        # 1. Input Encoder
        # Fourier Embedding for x, y, z, t (4 dims)
        self.num_freqs = 6
        self.fourier_dim = 4 * self.num_freqs * 2
        self.fourier_emb = FourierEmbedding(num_freqs=self.num_freqs, input_dim=4)

        # Input Projection: Fourier features + log_charge + auxiliary
        # Total input dims = fourier_dim + 2
        self.input_proj = nn.Sequential(
            nn.Linear(self.fourier_dim + 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.GELU(),
        )

        # 2. Backbone: Stack of DynGT Blocks
        self.blocks = nn.ModuleList(
            [
                DynGTBlock(
                    channels=self.hidden_channels,
                    heads=self.num_heads,
                    k_knn=self.k_knn,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. Global Pooling (Attention Pooling)
        # Learnable attention weights for aggregation
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_channels // 2, 1),
            )
        )

        # 4. Prediction Head
        self.head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 3),  # Predict vector (vx, vy, vz)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, N_Pulses, In_Channels).
                              In_Channels = 6: [x, y, z, time, log_charge, auxiliary]
        Returns:
            torch.Tensor: Predicted direction vectors (Batch, 3).
        """
        B, N, C = x.shape
        device = x.device

        # Flatten batch for PyTorch Geometric: (B*N, C)
        x_flat = x.view(B * N, C)

        # Create batch index vector: [0, 0, ..., 1, 1, ...]
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # Separate features
        # coords_time: [x, y, z, t] (indices 0-3)
        # feats: [log_charge, aux] (indices 4-5)
        coords_time = x_flat[:, :4]
        other_feats = x_flat[:, 4:]

        # Masking padding:
        # In data_processing, padding pulses have log_charge = -5.0.
        # We can create a mask to ignore them in pooling if needed,
        # but AttentionPooling usually handles this if the network learns to weight them 0.
        # For graph construction, they will be distant nodes or clustered together.

        # Apply Fourier Embedding
        pos_emb = self.fourier_emb(coords_time)  # (B*N, 48)

        # Concatenate and Project
        h = torch.cat([pos_emb, other_feats], dim=1)  # (B*N, 50)
        h = self.input_proj(h)  # (B*N, Hidden)

        # Pass through DynGT Blocks
        for block in self.blocks:
            h = block(h, batch_idx)

        # Global Pooling (Attention based)
        # Aggregates node features into graph features: (B, Hidden)
        h_graph = self.pool(h, batch_idx)

        # Prediction
        out = self.head(h_graph)  # (B, 3)

        return out
