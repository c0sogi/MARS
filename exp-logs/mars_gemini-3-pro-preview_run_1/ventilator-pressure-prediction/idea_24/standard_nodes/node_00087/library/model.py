import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Extracts features using multiple kernel sizes, concatenates, and projects
    to a unified model dimension.
    """

    def __init__(self, input_dim, model_dim, kernels=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()

        # We use model_dim as the channel size for each branch before projection
        # to ensure sufficient capacity to capture patterns at different scales.
        intermediate_dim = model_dim

        for k in kernels:
            # Padding = k // 2 ensures output length equals input length
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_dim, intermediate_dim, kernel_size=k, padding=k // 2
                    ),
                    nn.GELU(),
                )
            )

        # Projection layer to mix multi-scale features into the unified model dimension
        total_concat_dim = intermediate_dim * len(kernels)
        self.projection = nn.Linear(total_concat_dim, model_dim)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq) for Conv1d
        x = x.transpose(1, 2)

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)

        # (Batch, Total_Dim, Seq) -> (Batch, Seq, Total_Dim)
        out = out.transpose(1, 2)

        # Mix to model_dim
        out = self.projection(out)
        return out


class CompositeBlock(nn.Module):
    """
    Uniform-Capacity Composite Block.
    Features:
    - Curated Context Injection (concatenated to LSTM input)
    - Aligned Bi-LSTM (Input dim + Context -> 512 out)
    - Strict Identity Residuals (No LayerNorm)
    - Pointwise FFN
    """

    def __init__(self, model_dim, context_dim, lstm_hidden, expansion_factor, dropout):
        super().__init__()

        # LSTM input size is model_dim + context_dim
        # Output size is lstm_hidden * 2 (Bidirectional)
        # We ensure lstm_hidden * 2 == model_dim for identity residuals
        self.lstm = nn.LSTM(
            input_size=model_dim + context_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )

        # Pointwise Feed-Forward Network
        ffn_dim = model_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, model_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (Batch, Seq, model_dim)
        # context: (Batch, Seq, context_dim)

        # 1. Curated Context Injection
        lstm_in = torch.cat([x, context], dim=-1)

        # 2. Bi-LSTM
        lstm_out, _ = self.lstm(lstm_in)

        # 3. Strict Identity Residual 1
        x = x + self.dropout(lstm_out)

        # 4. FFN
        ffn_out = self.ffn(x)

        # 5. Strict Identity Residual 2
        x = x + self.dropout(ffn_out)

        return x


class VentilatorNet(nn.Module):
    """
    Uniform-Capacity Curated-Physics Composite Network.
    """

    def __init__(self):
        super().__init__()

        # --- Configuration ---
        # Input dim is 14 based on data_processing.py features (added u_out)
        input_dim = 14
        model_dim = Config.D_MODEL
        lstm_hidden = Config.LSTM_HIDDEN
        expansion = Config.EXPANSION_FACTOR
        dropout = Config.DROPOUT
        kernels = Config.STEM_KERNELS

        # Indices for Curated Context: R(2), C(3), R_u_in(5), vol_C(6)
        # Derived from library/data_processing.py
        self.context_indices = [2, 3, 5, 6]
        context_dim = len(self.context_indices)

        # --- Architecture ---

        # 1. Stem
        self.stem = MultiScaleStem(input_dim, model_dim, kernels)

        # 2. Backbone (Composite Blocks)
        self.blocks = nn.ModuleList()
        for _ in range(Config.NUM_LAYERS):
            self.blocks.append(
                CompositeBlock(model_dim, context_dim, lstm_hidden, expansion, dropout)
            )

        # 3. Heads
        # Auxiliary Head (Deep Supervision)
        self.aux_head = nn.Linear(model_dim, 1)

        # Final Head
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x, u_out=None):
        # x: (Batch, Seq, 13)

        # Extract Curated Context Vector
        context = x[:, :, self.context_indices]

        # Initial Feature Extraction
        h = self.stem(x)

        aux_pred = None

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            h = block(h, context)

            # Deep Supervision: Attach Aux Head after Block 2 (index 1)
            if i == 1:
                aux_pred = self.aux_head(h)

        # Final Prediction
        final_pred = self.head(h)

        # Remove feature dimension for regression output: (Batch, Seq, 1) -> (Batch, Seq)
        final_pred = final_pred.squeeze(-1)
        if aux_pred is not None:
            aux_pred = aux_pred.squeeze(-1)

        if self.training:
            return final_pred, aux_pred
        else:
            return final_pred
