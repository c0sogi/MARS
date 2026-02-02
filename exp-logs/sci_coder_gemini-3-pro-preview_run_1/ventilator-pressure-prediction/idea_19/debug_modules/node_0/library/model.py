import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Processes the input sequence with parallel convolutions of different kernel sizes
    to capture short-term and medium-term local dependencies.
    """

    def __init__(self, input_dim, d_model, kernels):
        super().__init__()
        # We maintain the input channel depth in the convolutions to preserve
        # raw signal fidelity before the projection.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernels
            ]
        )

        # The output dimension after concatenation is input_dim * number of kernels
        concat_dim = input_dim * len(kernels)
        self.proj = nn.Linear(concat_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        # Conv1d expects: (Batch, Features, Seq_Len)
        x_in = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            outs.append(conv(x_in))

        # Concatenate along the channel dimension
        # Shape: (Batch, Features * n_kernels, Seq_Len)
        cat = torch.cat(outs, dim=1)

        # Transpose back to sequence format: (Batch, Seq_Len, Features * n_kernels)
        cat = cat.transpose(1, 2)

        # Project to model dimension
        out = self.act(self.proj(cat))
        return out


class StrictIdentityBlock(nn.Module):
    """
    Strict-Identity Physics-Injected Composite Block.

    Key Innovations:
    1. Context Injection: Concatenates static physics features to the LSTM input.
    2. Aligned Dimensions: LSTM output width matches input width exactly.
    3. Strict Identity: Residual connection is a direct sum (no weights), ensuring
       unimpeded gradient flow.
    """

    def __init__(self, d_model, context_dim, lstm_hidden, expansion_factor, dropout):
        super().__init__()

        # Ensure dimensions align for strict identity residual
        if lstm_hidden * 2 != d_model:
            raise ValueError(
                f"Bidirectional LSTM output ({lstm_hidden * 2}) must match d_model ({d_model}) "
                "for strict identity mapping."
            )

        # 1. Physics-Injected Bi-LSTM
        # Input: d_model (from residual stream) + context_dim (physics features)
        self.lstm = nn.LSTM(
            input_size=d_model + context_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 2. Pointwise Feed-Forward Network (Channel Mixing)
        ffn_dim = d_model * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (Batch, Seq_Len, d_model)
        # context: (Batch, Seq_Len, context_dim)

        # --- Branch 1: Recurrent Processing with Context Injection ---
        # Concatenate the physics context to the latent representation
        lstm_input = torch.cat([x, context], dim=-1)

        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)

        # Strict Identity Residual 1
        # y = x + Dropout(LSTM(x + context))
        # No linear projection on x allows gradients to flow straight through.
        y = x + self.dropout(lstm_out)

        # --- Branch 2: Channel Mixing ---
        # Strict Identity Residual 2
        # z = y + Dropout(FFN(y))
        z = y + self.dropout(self.ffn(y))

        return z


class VentilatorModel(nn.Module):
    """
    Strict-Identity Physics-Injected Composite Network.

    Assembles the Multi-Scale Stem, a stack of Composite Blocks, and
    Deep Supervision heads.
    """

    def __init__(self, input_dim=None):
        super().__init__()

        # Determine input dimension if not provided explicitly
        if input_dim is None:
            # Based on library/data.py:
            # Base features (9): time_step, u_in, R, C, volume, R_u_in, vol_C, u_in_diff1, u_in_diff2
            # Lags: len(Config.LAGS)
            num_base = 9
            num_lags = len(Config.LAGS)
            input_dim = num_base + num_lags

        # Define indices for "static" physics features to be injected at every block.
        # Indices based on library/data.py feature_cols order:
        # R (2), C (3), R_u_in (5), vol_C (6)
        self.context_indices = [2, 3, 5, 6]
        context_dim = len(self.context_indices)

        # --- Architecture Components ---

        # 1. Stem
        self.stem = MultiScaleStem(
            input_dim=input_dim,
            d_model=Config.D_MODEL,
            kernels=Config.STEM_KERNELS,
        )

        # 2. Backbone (Composite Blocks)
        self.blocks = nn.ModuleList(
            [
                StrictIdentityBlock(
                    d_model=Config.D_MODEL,
                    context_dim=context_dim,
                    lstm_hidden=Config.LSTM_HIDDEN,
                    expansion_factor=Config.EXPANSION_FACTOR,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.N_BLOCKS)
            ]
        )

        # 3. Heads
        self.aux_head = nn.Linear(Config.D_MODEL, 1)
        self.head = nn.Linear(Config.D_MODEL, 1)

        # Auxiliary head position (0-indexed)
        self.aux_idx = Config.AUX_HEAD_IDX

    def forward(self, x):
        # x: (Batch, Seq_Len, input_dim)

        # Extract physics context for injection into blocks
        context = x[:, :, self.context_indices]

        # Initial embedding via Stem
        h = self.stem(x)

        aux_out = None

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            h = block(h, context)

            # Deep Supervision: Capture output after specific block
            if i == self.aux_idx:
                aux_out = self.aux_head(h)

        # Final prediction
        final_out = self.head(h)

        return final_out, aux_out
