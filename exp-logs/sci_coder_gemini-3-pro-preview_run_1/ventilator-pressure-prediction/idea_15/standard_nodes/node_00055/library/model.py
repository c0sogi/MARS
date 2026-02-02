import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Processes the input sequence with multiple kernel sizes to capture features
    at different temporal resolutions (noise vs trends).
    """

    def __init__(self, input_dim, hidden_dim, kernels=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            # Padding = k // 2 ensures output length matches input length (same padding)
            self.branches.append(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2)
            )

        # Project concatenated outputs back to hidden_dim to match backbone width
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernels), hidden_dim, kernel_size=1), nn.GELU()
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Dim) -> Transpose for Conv1d: (Batch, Dim, Seq)
        x = x.transpose(1, 2)

        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)

        out = self.project(concatenated)

        # Transpose back: (Batch, Seq, Dim)
        return out.transpose(1, 2)


class CompositeBlock(nn.Module):
    """
    Standard Composite Block with Deep Context Injection and Additive Residuals.

    Flow:
    1. Context Injection: Concat(Stream, Static Features)
    2. Temporal Mixing: Bi-LSTM
    3. Residual 1: Stream + Dropout(LSTM_Out)
    4. Channel Mixing: Pointwise FFN
    5. Residual 2: Stream + Dropout(FFN_Out)
    """

    def __init__(self, hidden_dim, static_dim, expansion_factor=2, dropout=0.1):
        super().__init__()

        # Temporal Mixing: Bi-LSTM
        # Input size increases due to context injection
        self.lstm = nn.LSTM(
            input_size=hidden_dim + static_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional sums to hidden_dim
            bidirectional=True,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Channel Mixing: Pointwise FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, stream, static_features):
        # stream: (Batch, Seq, Hidden)
        # static_features: (Batch, Seq, StaticDim)

        # 1. Deep Context Injection
        # Re-introduce physics constraints at every layer
        combined = torch.cat([stream, static_features], dim=-1)

        # 2. Temporal Mixing
        lstm_out, _ = self.lstm(combined)

        # 3. Additive Residual 1
        stream = stream + self.dropout1(lstm_out)

        # 4. Channel Mixing
        ffn_out = self.ffn(stream)

        # 5. Additive Residual 2
        stream = stream + self.dropout2(ffn_out)

        return stream


class VentilatorNet(nn.Module):
    """
    Deeply Supervised Physics-Injected Hybrid Network with Standard Residuals.

    Architecture:
    - Inputs -> MultiScale Stem
    - Block 1 (Composite)
    - Block 2 (Composite) -> Auxiliary Head (Deep Supervision)
    - Block 3 (Composite)
    - Block 4 (Composite) -> Final Head
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Identify Static Physics Features for Injection
        # We extract specific columns from the input to re-inject into blocks
        feature_map = config.get_feature_indices()
        target_static_feats = ["R", "C", "u_in_R", "vol_C"]

        # Filter to ensure features exist in the map
        self.static_indices = [
            feature_map[f] for f in target_static_feats if f in feature_map
        ]
        self.static_dim = len(self.static_indices)

        # 2. Stem
        self.stem = MultiScaleStem(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            kernels=config.STEM_KERNELS,
        )

        # 3. Backbone
        self.blocks = nn.ModuleList()
        for _ in range(config.NUM_BLOCKS):
            self.blocks.append(
                CompositeBlock(
                    hidden_dim=config.HIDDEN_DIM,
                    static_dim=self.static_dim,
                    expansion_factor=config.EXPANSION_FACTOR,
                    dropout=config.DROPOUT,
                )
            )

        # 4. Heads
        self.aux_head = nn.Linear(config.HIDDEN_DIM, 1)
        self.final_head = nn.Linear(config.HIDDEN_DIM, 1)

        # Attach auxiliary head after the 2nd block (index 1)
        self.aux_attach_idx = 1

    def forward(self, x, u_out=None, target=None):
        # x: (Batch, Seq, Feat)

        # Extract static features for deep injection
        static_feats = x[:, :, self.static_indices]

        # Pass through Stem
        stream = self.stem(x)

        aux_pred = None

        # Pass through Backbone
        for i, block in enumerate(self.blocks):
            stream = block(stream, static_feats)

            # Capture Auxiliary Prediction
            if i == self.aux_attach_idx:
                aux_pred = self.aux_head(stream).squeeze(-1)

        # Final Prediction
        final_pred = self.final_head(stream).squeeze(-1)

        output = {"prediction": final_pred}

        # Compute Loss if targets are provided (Training/Validation)
        if target is not None and u_out is not None:
            total_loss = self._compute_loss(final_pred, aux_pred, target, u_out)
            output["loss"] = total_loss

        return output

    def _compute_loss(self, final_pred, aux_pred, target, u_out):
        """
        Computes Weighted Masked L1 Loss.
        Masks out the expiratory phase (u_out=1) so only inspiratory phase is scored.
        """
        # Mask: 1.0 for Inspiratory (u_out=0), 0.0 for Expiratory
        mask = 1.0 - u_out

        # Safety for division
        mask_sum = mask.sum()
        if mask_sum < 1e-6:
            mask_sum = 1e-6

        # Main Head Loss
        loss_final = torch.abs(final_pred - target) * mask
        loss_final = loss_final.sum() / mask_sum

        # Auxiliary Head Loss
        loss_aux = torch.abs(aux_pred - target) * mask
        loss_aux = loss_aux.sum() / mask_sum

        # Composite Loss
        return loss_final + self.config.AUX_WEIGHT * loss_aux
