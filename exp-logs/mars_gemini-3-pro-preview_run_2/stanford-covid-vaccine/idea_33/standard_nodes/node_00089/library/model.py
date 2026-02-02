import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block with Dense Connections.

    Performs: Input -> [Conv1d (Dilated) -> ReLU -> Dropout] -> Output
    Returns: Concat([Input, Output], dim=1)
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseBlock, self).__init__()

        # Calculate padding to maintain sequence length: P = dilation * (kernel_size // 2)
        # Assumes stride=1 and odd kernel_size
        padding = dilation * (kernel_size // 2)

        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, In_Channels, Seq_Len)
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Dense connection: Concatenate input and output along channel dimension
        return torch.cat([x, out], dim=1)


class LatentInteraction(nn.Module):
    """
    Projects features, gathers partner features based on secondary structure, and fuses them.
    """

    def __init__(self, in_channels, latent_dim):
        super(LatentInteraction, self).__init__()
        self.proj = nn.Conv1d(in_channels, latent_dim, kernel_size=1)
        self.latent_dim = latent_dim

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Channels, Seq_Len)
            partner_indices: (Batch, Seq_Len) - indices of paired bases, -1 if unpaired
        """
        # 1. Project to latent dimension
        # Shape: (Batch, Latent_Dim, Seq_Len)
        h = self.proj(x)

        # Permute to (Batch, Seq_Len, Latent_Dim) for gathering
        h = h.permute(0, 2, 1)

        # 2. Gather Partner Features
        # Handle -1 indices by temporarily mapping them to 0, then masking result
        # Note: partner_indices are long ints

        # Create a safe index tensor (clone to avoid modifying input)
        safe_indices = partner_indices.clone()

        # Create mask: 1 where paired, 0 where unpaired (-1)
        mask = (safe_indices != -1).unsqueeze(-1).float()  # (Batch, Seq_Len, 1)

        # Replace -1 with 0 to prevent index out of bounds error during gather
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match latent dimension: (Batch, Seq_Len, Latent_Dim)
        expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim)

        # Gather along sequence dimension (dim 1)
        # h: (Batch, Seq_Len, Latent_Dim)
        h_partner = torch.gather(h, 1, expanded_indices)

        # Apply mask to zero out features gathered from dummy index 0 for unpaired bases
        h_partner = h_partner * mask

        # 3. Fuse: Concatenate local and partner features
        # Shape: (Batch, Seq_Len, Latent_Dim * 2)
        h_fused = torch.cat([h, h_partner], dim=2)

        return h_fused


class PFR_DN(nn.Module):
    """
    Projected Feedback Recurrent Dense Network (PFR-DN).

    Features:
    - Projected Recycling Mechanism
    - Dense Dilated TCN Backbone
    - Latent Structural Interaction
    - BiGRU Global Aggregation
    """

    def __init__(self):
        super(PFR_DN, self).__init__()

        # ----------------------------------------------------------------------
        # 1. Input Processing & Recycling
        # ----------------------------------------------------------------------
        # Projects the 5-channel regression feedback to a high-dim embedding
        self.recycling_proj = nn.Linear(Config.RECYCLING_DIM, Config.RECYCLING_PROJ_DIM)

        # Total input channels to backbone = Static Features + Recycling Embedding
        self.backbone_input_dim = Config.INPUT_DIM + Config.RECYCLING_PROJ_DIM

        # ----------------------------------------------------------------------
        # 2. Backbone: Dense Dilated TCN
        # ----------------------------------------------------------------------
        self.blocks = nn.ModuleList()
        current_dim = self.backbone_input_dim

        # Ensure kernel sizes list matches dilations length
        kernel_sizes = Config.KERNEL_SIZES
        if len(kernel_sizes) != len(Config.DILATIONS):
            # Fallback if lists don't match, though Config usually ensures they do
            kernel_sizes = [kernel_sizes[0]] * len(Config.DILATIONS)

        for kernel_size, dilation in zip(kernel_sizes, Config.DILATIONS):
            block = DenseBlock(
                in_channels=current_dim,
                growth_rate=Config.GROWTH_RATE,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.blocks.append(block)
            # In DenseNet, output of block is concat(input, new_features)
            current_dim += Config.GROWTH_RATE

        self.backbone_out_dim = current_dim

        # ----------------------------------------------------------------------
        # 3. Latent Structural Interaction
        # ----------------------------------------------------------------------
        self.interaction = LatentInteraction(
            in_channels=self.backbone_out_dim, latent_dim=Config.LATENT_DIM
        )
        # Interaction output is Latent_Dim * 2 (local + partner)
        self.interaction_out_dim = Config.LATENT_DIM * 2

        # ----------------------------------------------------------------------
        # 4. Global Aggregation (BiGRU)
        # ----------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.interaction_out_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            bidirectional=True,
            batch_first=True,
        )
        self.gru_out_dim = Config.RNN_HIDDEN_DIM * 2

        # ----------------------------------------------------------------------
        # 5. Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(self.gru_out_dim, Config.OUTPUT_DIM)

    def forward(self, x, partner_indices, recycling=None):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim=18) - Static features (Seq, Struct, Loop, PartnerID)
            partner_indices: (Batch, Seq_Len) - Indices of paired bases
            recycling: (Batch, Seq_Len, Output_Dim=5) - Previous predictions.
                       If None, initialized to zeros (Cold Start).

        Returns:
            out: (Batch, Seq_Len, Output_Dim=5)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Handle Recycling
        if recycling is None:
            recycling = torch.zeros(
                batch_size,
                seq_len,
                Config.RECYCLING_DIM,
                device=x.device,
                dtype=x.dtype,
            )

        # Project recycling: (B, L, 5) -> (B, L, 32)
        r_emb = self.recycling_proj(recycling)

        # Concatenate static features and recycling embedding
        # (B, L, 18) + (B, L, 32) -> (B, L, 50)
        combined_input = torch.cat([x, r_emb], dim=2)

        # Permute for Conv1d: (B, C, L)
        h = combined_input.permute(0, 2, 1)

        # 2. Backbone Pass
        for block in self.blocks:
            h = block(h)

        # h is now (B, Backbone_Out_Dim, L)

        # 3. Latent Interaction
        # Projects, permutes back to (B, L, Latent), gathers, and fuses
        # Returns (B, L, 128)
        h_interacted = self.interaction(h, partner_indices)

        # 4. Global Aggregation (BiGRU)
        # GRU expects (B, L, Input_Size)
        h_gru, _ = self.gru(h_interacted)

        # 5. Output Head
        out = self.head(h_gru)

        return out
