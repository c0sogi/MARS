import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleCNNStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Processes input with parallel kernels [3, 5, 7] to capture features at different resolutions.
    """

    def __init__(self, input_dim, hidden_dim, kernels=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Create parallel branches
        for k in kernels:
            # Padding = k // 2 ensures output length equals input length (same padding)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                )
            )

        # Projection to combine branches back to hidden_dim
        # Input channels = hidden_dim * number of branches
        self.projection = nn.Conv1d(
            hidden_dim * len(kernels), hidden_dim, kernel_size=1
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Permute for Conv1d: (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        concat = torch.cat(branch_outputs, dim=1)

        # Project and permute back
        out = self.projection(concat)
        out = out.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        return out


class CompositeBlock(nn.Module):
    """
    High-Capacity Composite Block.
    Consists of:
    1. Context-Injected Temporal Mixer (Bi-LSTM) with Physics Re-injection.
    2. Pointwise Channel Mixer (FFN).
    Uses additive residuals without LayerNorm to preserve signal amplitude.
    """

    def __init__(self, hidden_dim, physics_dim, dropout=0.1):
        super().__init__()

        # 1. Context-Injected Temporal Mixer
        # Input to LSTM is hidden_state + physics_features
        self.lstm = nn.LSTM(
            input_size=hidden_dim + physics_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional, so total output is hidden_dim
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_lstm = nn.Dropout(dropout)

        # 2. Pointwise Channel Mixer (FFN)
        # Expansion factor of 4 is standard for high capacity
        ffn_dim = hidden_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, physics_features):
        # x: (batch, seq_len, hidden_dim)
        # physics_features: (batch, seq_len, physics_dim)

        # --- Temporal Mixing ---
        # Concatenate physics features to the input at every step
        lstm_input = torch.cat([x, physics_features], dim=-1)

        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(lstm_input)

        # Residual Connection (No Norm)
        x = x + self.dropout_lstm(lstm_out)

        # --- Channel Mixing ---
        ffn_out = self.ffn(x)

        # Residual Connection (No Norm)
        x = x + self.dropout_ffn(ffn_out)

        return x


class VentilatorModel(nn.Module):
    """
    High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.aux_block_index = Config.AUX_BLOCK_INDEX
        self.dropout = Config.DROPOUT

        # Identify indices for physics features
        # We register this as a buffer so it moves to device automatically
        physics_indices = [
            Config.FEATURE_COLS.index(col) for col in Config.PHYSICS_COLS
        ]
        self.register_buffer(
            "physics_indices", torch.tensor(physics_indices, dtype=torch.long)
        )
        self.physics_dim = len(physics_indices)

        # 1. Stem
        self.stem = MultiScaleCNNStem(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernels=Config.CNN_KERNELS,
        )

        # 2. Backbone (Composite Blocks)
        self.blocks = nn.ModuleList(
            [
                CompositeBlock(
                    hidden_dim=self.hidden_dim,
                    physics_dim=self.physics_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. Heads
        self.aux_head = nn.Linear(self.hidden_dim, 1)
        self.final_head = nn.Linear(self.hidden_dim, 1)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Extract Physics Features for Re-injection
        # physics_features: (batch, seq_len, physics_dim)
        physics_features = torch.index_select(x, dim=-1, index=self.physics_indices)

        # Pass through Stem
        h = self.stem(x)

        aux_pred = None

        # Pass through Composite Blocks
        for i, block in enumerate(self.blocks):
            h = block(h, physics_features)

            # Capture Auxiliary Output
            if i == self.aux_block_index:
                aux_pred = self.aux_head(h).squeeze(-1)

        # Final Prediction
        final_pred = self.final_head(h).squeeze(-1)

        if self.training:
            return final_pred, aux_pred
        else:
            return final_pred
