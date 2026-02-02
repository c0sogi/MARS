import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A single dilated residual block for the TCN backbone.
    Structure: Conv1d (dilated) -> ReLU -> Dropout -> Conv1d (1x1) -> Residual
    """

    def __init__(self, in_channels, out_channels, dilation, kernel_size, dropout):
        super(DilatedResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)

        # If input and output channels differ, we need a projection for the residual
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return self.relu(out + residual)


class SF_DCN(nn.Module):
    """
    Selective-Feedback Dense-Context Network (SF-DCN).

    Features:
    1. Static Dense Dilated TCN Backbone.
    2. Selective Recycling: Masks unscored targets in the feedback loop.
    3. Partner-Aware Fusion: Explicitly gathers features from paired bases.
    4. Iterative Refinement: 2-pass mechanism.
    """

    def __init__(self, config: Config):
        super(SF_DCN, self).__init__()
        self.config = config

        # --- 1. Input Projection ---
        self.input_proj = nn.Conv1d(config.input_channels, config.hidden_dim, 1)

        # --- 2. Backbone (Dense Dilated TCN) ---
        self.blocks = nn.ModuleList()
        for d in config.dilations:
            self.blocks.append(
                DilatedResidualBlock(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    dilation=d,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
            )

        # Calculate size of dense features (concatenation of all block outputs)
        # Assuming we concatenate the output of each block
        self.dense_dim = config.hidden_dim * len(config.dilations)

        # --- 3. Latent Projection ---
        # Projects the dense backbone features to Z
        self.latent_proj = nn.Conv1d(self.dense_dim, config.latent_dim, 1)

        # --- 4. Feedback Module ---
        # Mask for selective recycling: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        # Scored: 0, 1, 3. Unscored: 2, 4.
        # Mask: [1, 1, 0, 1, 0]
        mask_values = [1.0 if i in config.scored_indices else 0.0 for i in range(5)]
        self.register_buffer(
            "feedback_mask",
            torch.tensor(mask_values, dtype=torch.float32).view(1, 5, 1),
        )

        self.feedback_proj = nn.Linear(5, config.feedback_dim)
        self.feedback_dropout = nn.Dropout(config.dropout)

        # --- 5. Fusion & Aggregation ---
        # Fusion Input:
        # Self: Z (latent_dim) + Feedback (feedback_dim)
        # Partner: Z (latent_dim) + Feedback (feedback_dim)
        # Total: 2 * (latent_dim + feedback_dim)
        fusion_dim = 2 * (config.latent_dim + config.feedback_dim)

        # BiGRU
        # Hidden size is set to input_dim // 2 to keep parameter count reasonable
        rnn_hidden = fusion_dim // 2
        self.rnn = nn.GRU(
            input_size=fusion_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- 6. Output Head ---
        self.head = nn.Linear(rnn_hidden * 2, 5)

    def _gather_partner(self, features, partner_indices):
        """
        Gathers features from the paired base.

        Args:
            features: (B, L, C)
            partner_indices: (B, L) - Indices of partners, -1 if unpaired.

        Returns:
            partner_features: (B, L, C) - Features of paired bases, 0 if unpaired.
        """
        batch_size, seq_len, channels = features.shape

        # Create a safe index tensor where -1 is replaced by 0
        # We will mask the result later
        safe_indices = partner_indices.clone()
        mask = (safe_indices != -1).unsqueeze(-1).float()  # (B, L, 1)
        safe_indices[safe_indices == -1] = 0

        # Gather
        # features is (B, L, C)
        # index needs to be expanded to (B, L, C)
        expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, channels)
        gathered = torch.gather(features, 1, expanded_indices.long())

        # Apply mask to zero out unpaired positions
        return gathered * mask

    def forward(self, x, partner_indices):
        """
        Args:
            x: (B, C, L) Input features
            partner_indices: (B, L) Partner indices

        Returns:
            y1: (B, L, 5) Prediction from Pass 1
            y2: (B, L, 5) Prediction from Pass 2
        """
        batch_size, _, seq_len = x.shape

        # --- 1. Static Backbone Pass ---
        h = self.input_proj(x)

        dense_features = []
        for block in self.blocks:
            h = block(h)
            dense_features.append(h)

        # Concatenate all block outputs (Dense Connection)
        # Shape: (B, dense_dim, L)
        h_dense = torch.cat(dense_features, dim=1)

        # Project to Latent Z: (B, latent_dim, L)
        z = self.latent_proj(h_dense)

        # Transpose for RNN/Gathering: (B, L, latent_dim)
        z_perm = z.permute(0, 2, 1)

        # --- 2. Iterative Refinement Loop ---

        # Initialize feedback with zeros
        y_preds = []
        current_feedback = torch.zeros(batch_size, seq_len, 5, device=x.device)

        # We run 2 passes
        # Pass 1: Feedback is 0
        # Pass 2: Feedback is Masked(y_pred_1)

        for i in range(2):
            # A. Prepare Feedback
            if i == 0:
                # Pass 1: Zero feedback
                e = torch.zeros(
                    batch_size, seq_len, self.config.feedback_dim, device=x.device
                )
            else:
                # Pass 2: Process previous prediction
                # 1. Detach gradients from previous pass
                prev_y = y_preds[-1].detach()

                # 2. Selective Recycling (Masking)
                # mask shape (1, 5, 1) -> transpose for (B, L, 5) multiplication: (1, 1, 5)
                mask = self.feedback_mask.permute(0, 2, 1)  # (1, 1, 5)
                masked_y = prev_y * mask

                # 3. Project & Dropout
                e = self.feedback_proj(masked_y)  # (B, L, feedback_dim)
                e = self.feedback_dropout(e)

            # B. Fusion (Self + Partner)
            # Self Vector: [Z_i, E_i]
            self_vec = torch.cat([z_perm, e], dim=-1)  # (B, L, latent + feedback)

            # Partner Vector: Gather([Z_j, E_j])
            partner_vec = self._gather_partner(self_vec, partner_indices)

            # Fuse
            rnn_input = torch.cat(
                [self_vec, partner_vec], dim=-1
            )  # (B, L, 2*(latent+feedback))

            # C. Aggregation (BiGRU)
            rnn_out, _ = self.rnn(rnn_input)

            # D. Output Head
            y_pred = self.head(rnn_out)  # (B, L, 5)
            y_preds.append(y_pred)

        return y_preds[0], y_preds[1]
