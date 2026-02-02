import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DistanceAwareInteractionModule(nn.Module):
    """
    Interaction module that incorporates relative distance information into the gating mechanism.
    Performs structural edge dropout during training to improve robustness.
    """

    def __init__(self, hidden_dim, dist_emb_dim, max_dist, edge_dropout=0.0):
        super(DistanceAwareInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dropout = edge_dropout

        # Relative Distance Embedding
        # Maps discrete distances [0, max_dist-1] to dense vectors
        self.distance_embedding = nn.Embedding(max_dist, dist_emb_dim)

        # Gating Network
        # Computes the trust gate z based on:
        # 1. Current state h_i
        # 2. Paired state h_j
        # 3. Relative distance embedding e_d
        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + dist_emb_dim, hidden_dim), nn.Sigmoid()
        )

        # Projection Network for the paired state h_j
        self.proj_net = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, pair_indices, distances):
        """
        Args:
            x (torch.Tensor): Hidden states (Batch, Seq_Len, Hidden_Dim)
            pair_indices (torch.Tensor): Indices of paired bases (Batch, Seq_Len)
            distances (torch.Tensor): Relative distances (Batch, Seq_Len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Gather paired hidden states h_j
        # pair_indices contains the index j for each position i.
        # We create a batch grid to gather correctly across the batch dimension.
        batch_idx = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Gather operation: x_paired[b, i] = x[b, pair_indices[b, i]]
        x_paired = x[batch_idx, pair_indices]

        # 2. Embed Distances
        # distances shape: (Batch, Seq_Len) -> (Batch, Seq_Len, Dist_Emb_Dim)
        d_emb = self.distance_embedding(distances)

        # 3. Compute Gate
        # Concatenate h_i, h_j, and e_d along the feature dimension
        concat = torch.cat([x, x_paired, d_emb], dim=-1)
        z = self.gate_net(concat)

        # 4. Compute Update Candidate
        u = self.proj_net(x_paired)

        # 5. Structural Edge Dropout
        # Randomly mask out interactions during training to prevent overfitting to specific structures
        if self.training and self.edge_dropout > 0:
            # Mask shape: (Batch, Seq_Len, 1) to broadcast over hidden dim
            # 1 = keep, 0 = drop
            keep_prob = 1.0 - self.edge_dropout
            mask = torch.bernoulli(
                torch.full((batch_size, seq_len, 1), keep_prob, device=x.device)
            )

            # Apply mask to the gate. If 0, z becomes 0, effectively skipping the update.
            z = z * mask

        # 6. Update State
        # h_new = h + z * u
        out = x + z * u
        return out


class DASR_BiGRU(nn.Module):
    """
    Distance-Augmented Structural-Refinement BiGRU.

    Architecture:
    1. Convolutional Stem: Projects sparse one-hot features to dense embeddings.
    2. Backbone (3 Blocks):
       - Block 1 & 2: BiGRU + Distance-Aware Interaction Module.
       - Block 3: BiGRU only (no interaction).
    3. Output Head: Linear projection to targets.
    """

    def __init__(self):
        super(DASR_BiGRU, self).__init__()

        # --- Convolutional Stem ---
        self.conv_stem = nn.Conv1d(
            in_channels=Config.NUM_NODE_FEATURES,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            padding=Config.CONV_KERNEL_SIZE // 2,
        )
        self.act = nn.GELU()
        self.dropout_stem = nn.Dropout(Config.DROPOUT)

        # --- Backbone Block 1 ---
        # BiGRU: Input 256 -> Output 384 (192*2)
        self.gru1 = nn.GRU(
            input_size=Config.CONV_FILTERS,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.interaction1 = DistanceAwareInteractionModule(
            hidden_dim=Config.HIDDEN_DIM,
            dist_emb_dim=Config.DISTANCE_EMBEDDING_DIM,
            max_dist=Config.MAX_DISTANCE,
            edge_dropout=Config.EDGE_DROPOUT,
        )
        self.norm1 = nn.LayerNorm(Config.HIDDEN_DIM)
        self.dropout1 = nn.Dropout(Config.DROPOUT)

        # --- Backbone Block 2 ---
        # BiGRU: Input 384 -> Output 384
        self.gru2 = nn.GRU(
            input_size=Config.HIDDEN_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.interaction2 = DistanceAwareInteractionModule(
            hidden_dim=Config.HIDDEN_DIM,
            dist_emb_dim=Config.DISTANCE_EMBEDDING_DIM,
            max_dist=Config.MAX_DISTANCE,
            edge_dropout=Config.EDGE_DROPOUT,
        )
        self.norm2 = nn.LayerNorm(Config.HIDDEN_DIM)
        self.dropout2 = nn.Dropout(Config.DROPOUT)

        # --- Backbone Block 3 ---
        # BiGRU: Input 384 -> Output 384
        # Note: No Interaction Module in the final block per strategy design.
        self.gru3 = nn.GRU(
            input_size=Config.HIDDEN_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(Config.HIDDEN_DIM)
        self.dropout3 = nn.Dropout(Config.DROPOUT)

        # --- Output Head ---
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, x, pair_indices, distances):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, 14)
            pair_indices (torch.Tensor): Structural pair indices (Batch, Seq_Len)
            distances (torch.Tensor): Relative distances (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 5)
        """
        # 1. Stem
        # Conv1d expects (Batch, Channels, Length)
        x = x.transpose(1, 2)
        x = self.conv_stem(x)
        x = self.act(x)
        x = self.dropout_stem(x)
        # Permute back to (Batch, Length, Channels) for GRU
        x = x.transpose(1, 2)

        # 2. Block 1
        out1, _ = self.gru1(x)
        out1 = self.interaction1(out1, pair_indices, distances)
        out1 = self.norm1(out1)
        out1 = self.dropout1(out1)

        # 3. Block 2
        out2, _ = self.gru2(out1)
        out2 = self.interaction2(out2, pair_indices, distances)
        out2 = self.norm2(out2)
        out2 = self.dropout2(out2)

        # 4. Block 3
        out3, _ = self.gru3(out2)
        # No interaction in final block
        out3 = self.norm3(out3)
        out3 = self.dropout3(out3)

        # 5. Head
        logits = self.head(out3)

        return logits
