import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem (Inception-style).
    Applies parallel convolutions with different kernel sizes, concatenates outputs,
    and projects to the model dimension.
    """

    def __init__(self, input_dim, d_model, kernels):
        super().__init__()
        # Parallel convolutions preserving temporal length (padding=k//2)
        # We keep the number of filters equal to input_dim for each branch
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k // 2)
                for k in kernels
            ]
        )

        # Concatenated dimension: input_dim * number of kernels
        concat_dim = input_dim * len(kernels)

        # Projection to d_model
        self.proj = nn.Linear(concat_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x shape: (Batch, Seq, Input_Dim)
        # Conv1d expects: (Batch, Input_Dim, Seq)
        x = x.transpose(1, 2)

        # Apply parallel convs
        outs = [conv(x) for conv in self.convs]

        # Concatenate along channel dimension
        x = torch.cat(outs, dim=1)

        # Transpose back: (Batch, Seq, Concat_Dim)
        x = x.transpose(1, 2)

        # Project and Activate
        x = self.act(self.proj(x))
        return x


class CompositeBlock(nn.Module):
    """
    Wide-State Identity-Residual Block with Deep Context Injection.
    """

    def __init__(self, d_model, context_dim, lstm_hidden, ffn_expansion, dropout):
        super().__init__()

        # 1. Deep Context Injection
        # The LSTM receives the current latent state concatenated with the static context features
        self.lstm_input_size = d_model + context_dim

        # 2. Wide-State Temporal Mixing (Bi-LSTM)
        # Hidden size is set such that output dim (2 * hidden) equals d_model
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Validation for Strict Identity Mapping
        if 2 * lstm_hidden != d_model:
            raise ValueError(
                f"Bi-LSTM output ({2*lstm_hidden}) must match d_model ({d_model}) "
                "for strict identity residual."
            )

        # 3. Pointwise Channel Mixing (FFN)
        ffn_dim = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        Args:
            x: Latent state (Batch, Seq, d_model)
            context: Static context features for injection (Batch, Seq, context_dim)
        """
        # --- Deep Context Injection & LSTM ---
        # Concatenate latent state with filtered static context
        x_in = torch.cat([x, context], dim=-1)

        # Bi-LSTM
        lstm_out, _ = self.lstm(x_in)

        # --- Strict Identity Residual 1 ---
        # No projection, direct addition
        x = x + self.dropout(lstm_out)

        # --- FFN & Strict Identity Residual 2 ---
        # No LayerNorm used to preserve absolute pressure magnitude
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class VentilatorModel(nn.Module):
    """
    Wide-State Identity-Residual Physics-Injected Composite Network.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Calculate input dimension based on configuration features
        # Continuous features + Binary features
        all_features = config.CONT_FEATURES + config.BINARY_FEATURES
        self.input_dim = len(all_features)

        # Identify indices for context features
        self.context_indices = [
            i for i, f in enumerate(all_features) if f in config.CONTEXT_FEATURES
        ]
        self.context_dim = len(self.context_indices)

        # --- Stem ---
        self.stem = MultiScaleStem(
            input_dim=self.input_dim,
            d_model=config.D_MODEL,
            kernels=config.STEM_KERNELS,
        )

        # --- Backbone ---
        self.blocks = nn.ModuleList()
        for _ in range(config.N_BLOCKS):
            self.blocks.append(
                CompositeBlock(
                    d_model=config.D_MODEL,
                    context_dim=self.context_dim,
                    lstm_hidden=config.LSTM_HIDDEN,
                    ffn_expansion=config.FFN_EXPANSION,
                    dropout=config.DROPOUT,
                )
            )

        # --- Heads ---
        # Auxiliary Head (Deep Supervision)
        self.aux_head = nn.Linear(config.D_MODEL, 1)

        # Final Head
        self.head = nn.Linear(config.D_MODEL, 1)

    def forward(self, x, u_out=None):
        """
        Args:
            x: Input features (Batch, Seq, Input_Dim)
            u_out: Not used directly in model logic, but passed for API consistency.
                   (Masking happens in loss function)
        Returns:
            final_pred: (Batch, Seq, 1)
            aux_pred: (Batch, Seq, 1) or None
        """
        # Extract static context for Deep Context Injection
        context = x[:, :, self.context_indices]

        # Pass through Stem
        h = self.stem(x)

        aux_pred = None

        # Pass through Blocks
        for i, block in enumerate(self.blocks):
            # Inject filtered context at every block
            h = block(h, context)

            # Deep Supervision: Capture output after Block 2 (index 1)
            if i == 1:
                aux_pred = self.aux_head(h)

        # Final Prediction
        final_pred = self.head(h)

        return final_pred, aux_pred
