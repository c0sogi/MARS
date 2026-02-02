import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WideBlock(nn.Module):
    """
    A reusable building block for Wide MLPs consisting of:
    Linear -> BatchNorm -> Activation -> Dropout.
    """

    def __init__(self, in_dim, out_dim, dropout_rate=0.2, activation=nn.ReLU()):
        super(WideBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (N, in_dim)
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AtomicStream(nn.Module):
    """
    Processes node-level atomic features (Identity, Spatial Context, Potential, Dist).
    Uses a Wide MLP to project features into a latent space and aggregates them
    using Dual Pooling (Global Mean + Global Max).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(AtomicStream, self).__init__()

        # Wide MLP Encoder
        # We use a few layers to capture non-linear interactions
        self.encoder = nn.Sequential(
            WideBlock(input_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
        )

        # Final linear projection to embedding space (no activation/bn here usually,
        # but the prompt implies output is an embedding. We'll project to output_dim).
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, max_atoms, input_dim)
            mask: (batch_size, max_atoms) - 1.0 for valid atoms, 0.0 for padding
        Returns:
            aggregated_embedding: (batch_size, output_dim * 2) -> Mean || Max
        """
        batch_size, max_atoms, feat_dim = x.shape

        # Flatten for processing: (batch_size * max_atoms, input_dim)
        x_flat = x.view(-1, feat_dim)

        # Pass through Wide MLP
        # Note: BatchNorm1d works on (N, C), so flattening is appropriate
        hidden = self.encoder(x_flat)

        # Project
        embedding = self.project(hidden)  # (batch_size * max_atoms, output_dim)

        # Reshape back
        embedding = embedding.view(batch_size, max_atoms, -1)  # (B, N, D)

        # Apply mask
        # Expand mask to (B, N, D)
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        embedding_masked = embedding * mask_expanded

        # 1. Global Mean Pooling
        # Sum valid embeddings and divide by number of valid atoms
        sum_pooled = torch.sum(embedding_masked, dim=1)  # (B, D)
        atom_counts = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean_pooled = sum_pooled / atom_counts

        # 2. Global Max Pooling
        # Set padded values to -inf before max
        # Create a large negative tensor
        neg_inf = torch.ones_like(embedding) * -1e9
        # Where mask is 1, keep embedding; where 0, use neg_inf
        embedding_for_max = torch.where(mask_expanded.bool(), embedding, neg_inf)
        max_pooled, _ = torch.max(embedding_for_max, dim=1)  # (B, D)

        # Concatenate
        return torch.cat([mean_pooled, max_pooled], dim=1)


class GlobalStream(nn.Module):
    """
    Processes macroscopic features (Lattice, Volume, Density, Stoichiometry, Total Atoms).
    Uses a High-Capacity MLP.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(GlobalStream, self).__init__()

        self.encoder = nn.Sequential(
            WideBlock(input_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            embedding: (batch_size, output_dim)
        """
        return self.encoder(x)


class RPA_WDS(nn.Module):
    """
    Robust Potential-Augmented Wide Deep Sets.
    Integrates Atomic and Global streams via Late Fusion.
    """

    def __init__(
        self,
        atomic_input_dim=Config.ATOMIC_INPUT_DIM,
        global_input_dim=Config.GLOBAL_INPUT_DIM,
        atomic_hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        global_hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(RPA_WDS, self).__init__()

        # Atomic Stream
        # The atomic stream outputs an embedding of size atomic_hidden_dim // 2
        # so that after concatenation (Mean+Max), the size is atomic_hidden_dim
        self.atomic_out_dim = atomic_hidden_dim // 2
        self.atomic_stream = AtomicStream(
            input_dim=atomic_input_dim,
            hidden_dim=atomic_hidden_dim,
            output_dim=self.atomic_out_dim,
            dropout_rate=dropout_rate,
        )

        # Global Stream
        self.global_stream = GlobalStream(
            input_dim=global_input_dim,
            hidden_dim=global_hidden_dim,
            output_dim=global_hidden_dim,  # Keep dimension high
            dropout_rate=dropout_rate,
        )

        # Fusion Head
        # Input: (Atomic_Mean + Atomic_Max) + Global
        fusion_input_dim = (self.atomic_out_dim * 2) + global_hidden_dim

        self.fusion_head = nn.Sequential(
            WideBlock(fusion_input_dim, fusion_hidden_dim, dropout_rate),
            WideBlock(fusion_hidden_dim, fusion_hidden_dim // 2, dropout_rate),
            nn.Linear(fusion_hidden_dim // 2, 2),  # Output: Formation Energy, Bandgap
        )

    def forward(self, atomic_x, global_x, mask):
        """
        Args:
            atomic_x: (B, N, atomic_feat_dim)
            global_x: (B, global_feat_dim)
            mask: (B, N)
        Returns:
            predictions: (B, 2)
        """
        # 1. Process Atomic Stream
        atomic_emb = self.atomic_stream(atomic_x, mask)  # (B, atomic_out_dim * 2)

        # 2. Process Global Stream
        global_emb = self.global_stream(global_x)  # (B, global_hidden_dim)

        # 3. Late Fusion
        combined = torch.cat([atomic_emb, global_emb], dim=1)

        # 4. Regression
        output = self.fusion_head(combined)

        return output
