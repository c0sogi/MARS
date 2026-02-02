import torch
import torch.nn as nn
from library.config import Config


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Stem).
    Extracts features using kernel sizes [3, 5, 7] and concatenates them.
    This captures both fine-grained signal noise and smoothed trend derivatives.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Split output_dim roughly equally among 3 branches
        dim1 = output_dim // 3
        dim2 = output_dim // 3
        dim3 = output_dim - dim1 - dim2

        self.conv3 = nn.Conv1d(input_dim, dim1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, dim2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, dim3, kernel_size=7, padding=3)

        # Activation (SiLU/Swish is generally robust for time series)
        self.act = nn.SiLU()

    def forward(self, x):
        # Input x: (Batch, Length, Features)
        # Conv1d expects: (Batch, Features, Length)
        x = x.transpose(1, 2)

        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        # Concatenate along feature dimension
        out = torch.cat([c3, c5, c7], dim=1)
        out = self.act(out)

        # Return to (Batch, Length, Features)
        return out.transpose(1, 2)


class CompositeBlock(nn.Module):
    """
    Wide-State Composite Block.

    Components:
    1. Deep Context Injection: Concatenates physics features to input.
    2. Wide-State Bi-LSTM: Output dim is 2 * hidden_dim (No compression).
    3. Projected Residual: Linear projection on identity path to match dims.
    4. Weight-Normalized High-Capacity FFN: Channel mixing with Weight Norm.
    """

    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()

        # The LSTM output dimension is 2 * hidden_dim (Bidirectional)
        # We explicitly do NOT compress this back to hidden_dim.
        self.lstm_output_dim = 2 * hidden_dim

        # 1. Wide-State Bi-LSTM
        # Input is previous state + context features
        self.lstm = nn.LSTM(
            input_size=input_dim + context_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 2. Projected Residual 1
        # Projects the input 'x' to match the LSTM output dimension
        self.proj_res1 = nn.Linear(input_dim, self.lstm_output_dim)

        # 3. FFN (Channel Mixing) - Removed Weight Norm (Cite solution_lesson_node_00062)
        # Expansion factor reduced to 2x (Cite solution_lesson_node_00052)
        expansion_dim = self.lstm_output_dim * Config.FFN_EXPANSION_FACTOR

        self.ffn = nn.Sequential(
            nn.Linear(self.lstm_output_dim, expansion_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(expansion_dim, self.lstm_output_dim),
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x, context):
        # x: (B, L, input_dim)
        # context: (B, L, context_dim)

        # Deep Context Injection
        lstm_in = torch.cat([x, context], dim=-1)

        # Wide-State Temporal Mixing
        # lstm_out: (B, L, 2*hidden_dim)
        lstm_out, _ = self.lstm(lstm_in)

        # Projected Residual 1
        # Align input x to lstm output dim
        res1 = self.proj_res1(x)
        mid = lstm_out + res1

        # High-Capacity Channel Mixing (FFN)
        ffn_out = self.ffn(mid)

        # Residual 2 (Additive around FFN)
        out = mid + self.dropout(ffn_out)

        return out


class WideStateNet(nn.Module):
    """
    Wide-State Weight-Normalized Physics-Injected Composite Network.

    Structure:
    - Multi-Scale Stem
    - Stack of Composite Blocks (Wide State)
    - Deep Supervision (Auxiliary Head)
    - Final Regression Head
    """

    def __init__(self, input_dim, feature_names):
        super().__init__()
        self.feature_names = feature_names

        # Identify Physics Feature Indices for Injection
        # We look for specific physics-based features to inject at every layer
        target_features = ["R", "C", "u_in_R", "vol_C"]
        self.ctx_indices = []
        for feat in target_features:
            if feat in feature_names:
                self.ctx_indices.append(feature_names.index(feat))

        self.context_dim = len(self.ctx_indices)

        # Architecture Hyperparameters
        hidden_dim = Config.HIDDEN_DIM
        num_layers = Config.LSTM_LAYERS

        # Stem
        self.stem = MultiScaleCNN(input_dim, hidden_dim)

        # Backbone
        self.blocks = nn.ModuleList()

        # Block 1:
        # Input: hidden_dim (from stem)
        # Output: 2 * hidden_dim (Wide State expansion)
        self.blocks.append(CompositeBlock(hidden_dim, hidden_dim, self.context_dim))

        # Subsequent Blocks:
        # Input: 2 * hidden_dim
        # Output: 2 * hidden_dim
        # We start loop from 1 because we already added Block 1
        for _ in range(num_layers - 1):
            self.blocks.append(
                CompositeBlock(2 * hidden_dim, hidden_dim, self.context_dim)
            )

        # Deep Supervision Head (Attached after Block 2, i.e., index 1)
        if Config.USE_AUXILIARY_HEAD:
            self.aux_head = nn.Linear(2 * hidden_dim, 1)
        else:
            self.aux_head = None

        # Final Head
        self.head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, u_out=None):
        # x: (B, L, F)
        # u_out: (B, L) - Optional, not used in architecture but kept for API consistency

        # Extract Physics Context
        if self.context_dim > 0:
            context = x[:, :, self.ctx_indices]
        else:
            # Empty tensor if no context features found (fallback)
            context = x.new_empty(x.size(0), x.size(1), 0)

        # Stem
        h = self.stem(x)

        aux_pred = None

        # Blocks
        for i, block in enumerate(self.blocks):
            h = block(h, context)

            # Auxiliary Head after Block 2 (index 1)
            if i == 1 and self.aux_head is not None:
                aux_pred = self.aux_head(h)

        # Final Prediction
        final_pred = self.head(h)

        # Squeeze last dim: (B, L, 1) -> (B, L)
        final_pred = final_pred.squeeze(-1)
        if aux_pred is not None:
            aux_pred = aux_pred.squeeze(-1)

        return final_pred, aux_pred
