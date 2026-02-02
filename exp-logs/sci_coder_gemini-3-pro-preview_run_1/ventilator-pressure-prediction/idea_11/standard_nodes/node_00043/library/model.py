import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from library.config import Config


class InceptionStem(nn.Module):
    """
    Multi-scale feature extraction using parallel 1D convolutions.
    Captures both fine-grained signal noise and smoothed trend derivatives.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        super().__init__()
        self.convs = nn.ModuleList()

        # Calculate output channels per branch to sum up roughly to out_channels
        # We will project the concatenated output to exact out_channels afterwards
        branch_channels = out_channels // len(kernel_sizes)

        for k in kernel_sizes:
            # Padding = 'same' ensures output length matches input length
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=k,
                    padding="same",
                )
            )

        concat_channels = branch_channels * len(kernel_sizes)
        self.project = nn.Conv1d(concat_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Features) -> (Batch, Features, Seq_Len) for Conv1d
        x = x.transpose(1, 2)

        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)
        x = self.project(x)
        x = self.bn(x)
        x = self.activation(x)

        # (Batch, Features, Seq_Len) -> (Batch, Seq_Len, Features)
        return x.transpose(1, 2)


class PointwiseFFN(nn.Module):
    """
    Pointwise Feed-Forward Network for channel mixing.
    Decouples temporal mixing (LSTM) from channel mixing.
    Structure: Linear -> GELU -> Linear
    """

    def __init__(self, hidden_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompositeBlock(nn.Module):
    """
    A composite block containing:
    1. Context-Injected Bidirectional LSTM (Temporal Mixing)
    2. Pointwise FFN (Channel Mixing)
    3. Residual connections and Normalization
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, context_dim: int, dropout: float = 0.1
    ):
        super().__init__()

        # 1. Context-Injected Recurrent Sub-layer
        # Input to LSTM is concatenation of previous hidden state and physics context
        self.lstm_input_dim = input_dim + context_dim

        # Bidirectional LSTM
        # We set hidden_size to hidden_dim // 2 so that output size (fwd + bwd) is hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Projection layer in case input_dim != hidden_dim (though usually they match in backbone)
        self.lstm_proj = (
            nn.Linear(hidden_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # 2. Pointwise FFN Sub-layer
        self.ffn = PointwiseFFN(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq_Len, Hidden_Dim)
        context: (Batch, Seq_Len, Context_Dim) - Static features repeated/expanded
        """
        # --- Sub-layer 1: Context-Injected LSTM ---
        residual = x

        # Concatenate context
        lstm_input = torch.cat([x, context], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Projection (if needed) and Dropout
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Additive Residual + Norm
        x = self.ln1(residual + lstm_out)

        # --- Sub-layer 2: Pointwise FFN ---
        residual = x
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)  # Re-use dropout

        # Additive Residual + Norm
        x = self.ln2(residual + ffn_out)

        return x


class VentilatorModel(nn.Module):
    """
    Deeply Supervised Physics-Injected Hybrid CNN-LSTM-FFN.
    """

    def __init__(self, config: Config = Config):
        super().__init__()
        self.config = config

        # Identify indices of context features
        self.context_indices = [
            config.INPUT_FEATURES.index(f) for f in config.CONTEXT_FEATURES
        ]
        self.num_context = len(config.CONTEXT_FEATURES)
        self.num_input = len(config.INPUT_FEATURES)

        # --- Stem ---
        self.stem = InceptionStem(
            in_channels=self.num_input,
            out_channels=config.HIDDEN_DIM,
            kernel_sizes=config.STEM_KERNEL_SIZES,
        )

        # --- Backbone ---
        self.blocks = nn.ModuleList()
        for _ in range(config.NUM_BLOCKS):
            self.blocks.append(
                CompositeBlock(
                    input_dim=config.HIDDEN_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    context_dim=self.num_context,
                    dropout=config.DROPOUT,
                )
            )

        # --- Heads ---
        self.aux_head = nn.Linear(config.HIDDEN_DIM, 1)
        self.final_head = nn.Linear(config.HIDDEN_DIM, 1)

        self.aux_idx = config.AUX_BLOCK_INDEX

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (Batch, Seq_Len, Num_Features)

        Returns:
            final_pred: (Batch, Seq_Len)
            aux_pred: (Batch, Seq_Len) or None (if inference only, though we return both for consistency)
        """
        # 1. Extract Context Features
        # x is (B, L, F). We slice the context features.
        context = x[:, :, self.context_indices]

        # 2. Stem
        x = self.stem(x)

        # 3. Backbone
        aux_pred = None

        for i, block in enumerate(self.blocks):
            x = block(x, context)

            # Deep Supervision
            if i == self.aux_idx:
                aux_pred = self.aux_head(x).squeeze(-1)

        # 4. Final Head
        final_pred = self.final_head(x).squeeze(-1)

        return final_pred, aux_pred
