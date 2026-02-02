import torch
import torch.nn as nn
from library.config import Config


class DilatedDenseLayer(nn.Module):
    """
    Single-Layer Dilated Residual Block adapted for DenseNet connectivity.
    Applies Conv1d -> ReLU -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        # Calculate padding to maintain sequence length (assuming odd kernel_size)
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return out


class SR_DCN(nn.Module):
    """
    Stabilized Recurrent Dense-Context Network (SR-DCN).

    Architecture:
    1. Input: Static Features + Dynamic Recycling Channels.
    2. Backbone: Dense Dilated TCN (Dense connections across exponentially dilated layers).
    3. Interaction: Latent projection + Symmetric Partner Gathering (Partner-Aware).
    4. Aggregation: BiGRU for global context.
    5. Head: Linear projection to targets.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Input Configuration
        # =====================================================================
        # Static Channels:
        #   Sequence (4) + Structure (3) + Loop (7) + PartnerID (4) = 18
        self.static_channels = 18
        # Dynamic Channels:
        #   Recycling (5 targets)
        self.recycling_channels = Config.RECYCLING_CHANNELS

        self.in_channels = self.static_channels + self.recycling_channels

        # =====================================================================
        # 2. Dense Dilated Backbone
        # =====================================================================
        self.layers = nn.ModuleList()
        current_dim = self.in_channels

        # Stack layers with exponential dilation
        for d in Config.DILATIONS:
            layer = DilatedDenseLayer(
                in_channels=current_dim,
                growth_rate=Config.GROWTH_RATE,
                kernel_size=Config.KERNEL_SIZE,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.layers.append(layer)
            # In DenseNet, next layer input grows by growth_rate
            current_dim += Config.GROWTH_RATE

        self.backbone_out_dim = current_dim

        # =====================================================================
        # 3. Latent Structural Interaction
        # =====================================================================
        self.hidden_dim = Config.HIDDEN_DIM

        # Project high-dimensional dense history to compact latent dim
        self.projector = nn.Conv1d(
            self.backbone_out_dim, self.hidden_dim, kernel_size=1
        )

        # =====================================================================
        # 4. Global Aggregation (BiGRU)
        # =====================================================================
        # Input: Local Latent (64) + Partner Latent (64) = 128
        gru_input_dim = self.hidden_dim * 2

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.hidden_dim,  # Output dim will be hidden_dim * 2 (Bidirectional)
            batch_first=True,
            bidirectional=True,
        )

        # =====================================================================
        # 5. Output Head
        # =====================================================================
        # BiGRU output is (B, L, hidden_dim * 2)
        self.head = nn.Linear(self.hidden_dim * 2, len(Config.TARGET_COLS))

    def forward(self, inputs, partner_indices, recycling=None):
        """
        Args:
            inputs (torch.Tensor): Static features (B, L, 18).
            partner_indices (torch.Tensor): Partner indices (B, L).
            recycling (torch.Tensor, optional): Recycling features (B, L, 5).
                                               Defaults to zeros if None.

        Returns:
            torch.Tensor: Predictions (B, L, 5).
        """
        B, L, _ = inputs.shape
        device = inputs.device

        # --- 1. Input Preparation ---
        if recycling is None:
            recycling = torch.zeros(B, L, self.recycling_channels, device=device)

        # Concatenate static and recycling features
        # Shape: (B, L, 23)
        x = torch.cat([inputs, recycling], dim=2)
        # Permute for Conv1d: (B, 23, L)
        x = x.permute(0, 2, 1)

        # --- 2. Dense Backbone ---
        # Maintain a list of feature maps for dense connectivity
        features = [x]

        for layer in self.layers:
            # Concatenate all previous outputs to form input for current layer
            in_feat = torch.cat(features, dim=1)
            out_feat = layer(in_feat)
            features.append(out_feat)

        # Final representation is concatenation of all layers
        # Shape: (B, Total_Channels, L)
        backbone_out = torch.cat(features, dim=1)

        # --- 3. Latent Structural Interaction ---
        # Project to latent space
        # Shape: (B, 64, L)
        latent = self.projector(backbone_out)
        # Permute back to (B, L, 64) for gathering and RNN
        latent = latent.permute(0, 2, 1)

        # Gather Partner Features
        # Create mask for valid partners (index != -1)
        valid_mask = (partner_indices != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Replace -1 with 0 for safe gathering indices
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match feature dimension: (B, L, 64)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        # Gather features: output[b, i, c] = input[b, index[b, i, c], c]
        partner_features = torch.gather(latent, 1, gather_indices)

        # Apply Null-Masking (zero out features for unpaired bases)
        partner_features = partner_features * valid_mask

        # Fuse Local and Partner features
        # Shape: (B, L, 128)
        fused = torch.cat([latent, partner_features], dim=2)

        # --- 4. Global Aggregation ---
        # Shape: (B, L, 128)
        gru_out, _ = self.gru(fused)

        # --- 5. Output Head ---
        # Shape: (B, L, 5)
        logits = self.head(gru_out)

        return logits
