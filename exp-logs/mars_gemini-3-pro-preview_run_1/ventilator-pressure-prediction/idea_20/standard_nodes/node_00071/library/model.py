import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Processes the input sequence with parallel convolutions of different kernel sizes
    to capture features at various temporal resolutions.
    """

    def __init__(self, input_dim, model_dim, kernels):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            # Padding to maintain sequence length (Same padding)
            pad = k // 2
            self.convs.append(
                nn.Conv1d(
                    input_dim, model_dim // len(kernels), kernel_size=k, padding=pad
                )
            )

        # Projection to exact model_dim in case integer division leaves a remainder
        concat_dim = (model_dim // len(kernels)) * len(kernels)
        self.projection = nn.Linear(concat_dim, model_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (N, L, Input_Dim) -> Permute to (N, Input_Dim, L) for Conv1d
        x = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            outs.append(conv(x))

        # Concatenate along channel dimension: (N, C_total, L)
        x = torch.cat(outs, dim=1)

        # Permute back to (N, L, C_total)
        x = x.transpose(1, 2)

        # Project and Activate
        x = self.projection(x)
        x = self.activation(x)
        return x


class CompositeBlock(nn.Module):
    """
    Curated-Identity Composite Block.
    Features:
    1. Curated Context Injection: Concatenates static features to LSTM input.
    2. Aligned Wide-State Bi-LSTM: Hidden size matches model dimension.
    3. Strict Identity Residuals: No linear weights on skip connections.
    4. Pointwise Channel Mixing (FFN).
    5. No Layer Normalization.
    """

    def __init__(self, model_dim, lstm_hidden, dropout, static_dim=2):
        super().__init__()

        # 1. Curated Context Injection
        # Input to LSTM is Model_Dim + Static_Dim (R, C)
        lstm_input_dim = model_dim + static_dim

        # 2. Bi-LSTM
        # Output dim = 2 * lstm_hidden. We ensure this equals model_dim in Config.
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 3. FFN (Pointwise)
        # Expansion factor 2
        ffn_dim = model_dim * Config.FFN_EXPANSION
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, model_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static):
        # x: (N, L, D)
        # static: (N, S) -> (N, 2)

        # --- Sub-Block 1: Context-Injected LSTM with Identity Residual ---

        # Expand static features to sequence length: (N, L, 2)
        seq_len = x.size(1)
        static_expanded = static.unsqueeze(1).expand(-1, seq_len, -1)

        # Curated Injection: Concatenate x and static
        lstm_input = torch.cat([x, static_expanded], dim=-1)

        # LSTM Processing
        lstm_out, _ = self.lstm(lstm_input)

        # Strict Identity Residual 1
        # y = x + Dropout(LSTM(concat(x, static)))
        y = x + self.dropout(lstm_out)

        # --- Sub-Block 2: FFN with Identity Residual ---

        # FFN Processing
        ffn_out = self.ffn(y)

        # Strict Identity Residual 2
        # z = y + Dropout(FFN(y))
        z = y + self.dropout(ffn_out)

        return z


class CuratedIdentityNet(nn.Module):
    """
    Full Architecture: Curated-Identity Physics-Composite Network.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        input_dim = Config.INPUT_DIM
        model_dim = Config.MODEL_DIM
        kernels = Config.CNN_KERNELS
        num_blocks = Config.NUM_BLOCKS
        lstm_hidden = Config.LSTM_HIDDEN
        dropout = Config.DROPOUT

        # Stem
        self.stem = MultiScaleStem(input_dim, model_dim, kernels)

        # Backbone
        self.blocks = nn.ModuleList(
            [
                CompositeBlock(
                    model_dim=model_dim,
                    lstm_hidden=lstm_hidden,
                    dropout=dropout,
                    static_dim=2,  # R and C
                )
                for _ in range(num_blocks)
            ]
        )

        # Heads
        self.head = nn.Linear(model_dim, 1)
        self.aux_head = nn.Linear(model_dim, 1)

        self.aux_index = Config.AUX_BLOCK_INDEX

    def forward(self, x, static, u_out=None):
        # x: (N, L, Input_Dim)
        # static: (N, 2)

        # Stem
        x = self.stem(x)

        aux_pred = None

        # Backbone
        for i, block in enumerate(self.blocks):
            x = block(x, static)

            # Deep Supervision
            if i == self.aux_index:
                aux_pred = self.aux_head(x)

        # Final Prediction
        final_pred = self.head(x)

        return final_pred, aux_pred
