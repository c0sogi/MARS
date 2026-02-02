import torch
import torch.nn as nn
from library.config import Config


class MultiScaleCNN(nn.Module):
    """
    Applies parallel 1D convolutions with different kernel sizes to capture
    features at multiple temporal scales.
    """

    def __init__(self, input_dim, filters, kernels):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=filters,
                    kernel_size=k,
                    padding=k // 2,  # 'Same' padding
                    padding_mode="zeros",
                )
                for k in kernels
            ]
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        # Apply each conv and concatenate along the channel dimension
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)


class PhysicsInjectedNet(nn.Module):
    """
    Physics-Injected Residual Multi-Scale CNN-LSTM.
    Injects physical context (R, C, interactions) at every layer of the LSTM backbone
    to preserve physical constraints throughout the network depth.
    """

    def __init__(self):
        super().__init__()
        self.config = Config

        # 1. Resolve Feature Indices for Context Injection
        # We need to know which columns in the input tensor correspond to the context features.
        self.feature_cols = self.config.FEATURE_COLS
        self.context_cols = self.config.CONTEXT_FEATURES

        # Create a list of indices for slicing
        self.context_indices = [self.feature_cols.index(c) for c in self.context_cols]
        self.context_dim = len(self.context_cols)

        # Total input dimension
        input_dim = len(self.feature_cols)

        # 2. Stem: Multi-Scale CNN
        self.stem = MultiScaleCNN(
            input_dim=input_dim,
            filters=self.config.CNN_FILTERS,
            kernels=self.config.CNN_KERNELS,
        )
        self.stem_act = nn.GELU()

        # Calculate output dimension of the stem
        # filters * number of kernels
        stem_out_dim = self.config.CNN_FILTERS * len(self.config.CNN_KERNELS)

        # 3. Backbone: Deep Context-Injected Bi-LSTM
        self.lstm_layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = stem_out_dim
        hidden_dim = self.config.LSTM_HIDDEN_DIM
        bidirectional = self.config.BIDIRECTIONAL
        num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * num_directions

        for _ in range(self.config.LSTM_LAYERS):
            # The input to the LSTM is the previous hidden state + context features
            lstm_input_dim = current_dim + self.context_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )

            # Residual Projection:
            # If the input dimension (current_dim) doesn't match the LSTM output dimension,
            # we project the residual path to match.
            if current_dim != lstm_out_dim:
                self.projections.append(nn.Linear(current_dim, lstm_out_dim))
            else:
                self.projections.append(nn.Identity())

            # Dropout applied to the residual branch
            self.dropouts.append(nn.Dropout(self.config.DROPOUT))

            # Update current dimension for the next layer
            current_dim = lstm_out_dim

        # 4. Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, N_Features)
        Returns:
            Tensor of shape (Batch, Seq_Len)
        """
        # Extract Static/Physical Context Features
        # Shape: (Batch, Seq_Len, Context_Dim)
        context = x[:, :, self.context_indices]

        # --- Stem ---
        # Permute for Conv1d: (Batch, Feat, Seq)
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.stem_act(h)
        # Permute back: (Batch, Seq, Feat)
        h = h.transpose(1, 2)

        # --- Backbone ---
        for i in range(len(self.lstm_layers)):
            # 1. Context Injection
            # Concatenate current hidden state with context features
            # h: (B, S, H_curr), context: (B, S, C) -> (B, S, H_curr + C)
            lstm_in = torch.cat([h, context], dim=2)

            # 2. LSTM Processing
            # out: (B, S, H_next)
            out, _ = self.lstm_layers[i](lstm_in)

            # 3. Residual Connection
            # Path A: Project previous hidden state to match dimensions
            res_path = self.projections[i](h)

            # Path B: Dropout on the computed features
            out_path = self.dropouts[i](out)

            # Add
            h = res_path + out_path

        # --- Head ---
        # h: (Batch, Seq, Hidden_Final)
        preds = self.head(h)

        # Squeeze last dimension: (Batch, Seq, 1) -> (Batch, Seq)
        return preds.squeeze(-1)
