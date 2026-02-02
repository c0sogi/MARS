import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedConvBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SR_DCN(nn.Module):
    """
    Stabilized Recurrent Dense-Context Network (SR-DCN).
    """

    def __init__(self):
        super(SR_DCN, self).__init__()

        # ==============================
        # Hyperparameters & Dimensions
        # ==============================
        self.input_channels = Config.INPUT_CHANNELS  # 23 (18 static + 5 recycling)
        self.hidden_dim = Config.HIDDEN_DIM  # 64
        self.growth_rate = Config.GROWTH_RATE  # 64
        self.dilation_rates = Config.DILATION_RATES  # [1, 2, 4, 8, 16, 32]
        self.dropout_rate = Config.DROPOUT  # 0.1
        self.latent_dim = Config.LATENT_DIM  # 64

        # ==============================
        # 1. Stem
        # ==============================
        self.stem = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=1)

        # ==============================
        # 2. Dense Dilated Backbone
        # ==============================
        self.blocks = nn.ModuleList()
        current_channels = self.hidden_dim

        for dilation in self.dilation_rates:
            block = DilatedConvBlock(
                in_channels=current_channels,
                out_channels=self.growth_rate,
                dilation=dilation,
                dropout=self.dropout_rate,
            )
            self.blocks.append(block)
            # In a DenseNet pattern, the next block receives all previous features concatenated
            current_channels += self.growth_rate

        self.total_backbone_channels = current_channels

        # ==============================
        # 3. Latent Structural Interaction
        # ==============================
        # Project unified history to latent dimension (Linear 1x1)
        self.latent_proj = nn.Conv1d(
            self.total_backbone_channels, self.latent_dim, kernel_size=1
        )

        # ==============================
        # 4. Global Aggregation (BiGRU)
        # ==============================
        # Input: Local Latent (64) + Partner Latent (64) = 128
        self.gru_input_dim = self.latent_dim * 2
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=self.gru_input_dim // 2,  # 64 per direction -> 128 total
            bidirectional=True,
            batch_first=True,
        )

        # ==============================
        # 5. Output Head
        # ==============================
        # BiGRU output is hidden_size * 2 = 128
        self.head = nn.Linear(self.gru_input_dim, 5)

    def forward(self, x, recycling, partner_indices):
        """
        Args:
            x (torch.Tensor): Static features. Shape (Batch, Seq_Len, 18).
            recycling (torch.Tensor): Recycling channels (previous predictions). Shape (Batch, Seq_Len, 5).
            partner_indices (torch.Tensor): Indices of paired bases (-1 if unpaired). Shape (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions. Shape (Batch, Seq_Len, 5).
        """
        # ---------------------------------------------------------
        # 1. Input Preparation
        # ---------------------------------------------------------
        # Concatenate static and recycling features along channel dimension
        # x: (B, L, 18), recycling: (B, L, 5) -> (B, L, 23)
        x = torch.cat([x, recycling], dim=2)

        # Permute for Conv1d: (B, C, L)
        x = x.permute(0, 2, 1)

        # ---------------------------------------------------------
        # 2. Backbone (Dense Dilated TCN)
        # ---------------------------------------------------------
        stem_out = self.stem(x)

        # Initialize features list with stem output
        features = [stem_out]

        for block in self.blocks:
            # Dense Connection: Concatenate all previous features
            in_feat = torch.cat(features, dim=1)
            out = block(in_feat)
            features.append(out)

        # Unified History: Concatenate all block outputs + stem
        backbone_out = torch.cat(features, dim=1)  # (B, Total_Channels, L)

        # ---------------------------------------------------------
        # 3. Latent Structural Interaction
        # ---------------------------------------------------------
        # Project to latent dimension
        latent = self.latent_proj(backbone_out)  # (B, 64, L)

        # Permute back to (B, L, C) for gathering and RNN
        latent = latent.permute(0, 2, 1)  # (B, L, 64)

        # --- Partner Gathering Logic ---
        # partner_indices: (B, L)

        # Create mask for valid partners (where index != -1)
        valid_mask = (partner_indices != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Replace -1 with 0 for safe gathering (will be masked out later)
        # We clone to avoid modifying the input tensor in place
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match latent dimension: (B, L, 64)
        # We need to gather from dim 1 (Seq_Len).
        safe_indices_expanded = safe_indices.unsqueeze(-1).expand(
            -1, -1, self.latent_dim
        )

        # Gather partner features
        # latent is (B, L, C). We use gather on dim 1.
        partner_latent = torch.gather(latent, 1, safe_indices_expanded)

        # Apply Null-Masking: Zero out features gathered from index 0 if the original index was -1
        partner_latent = partner_latent * valid_mask

        # Fusion: Concatenate Local + Partner
        # (B, L, 64) + (B, L, 64) -> (B, L, 128)
        fused = torch.cat([latent, partner_latent], dim=2)

        # ---------------------------------------------------------
        # 4. Global Aggregation & Head
        # ---------------------------------------------------------
        # BiGRU
        gru_out, _ = self.gru(fused)  # (B, L, 128)

        # Linear Head
        logits = self.head(gru_out)  # (B, L, 5)

        return logits
