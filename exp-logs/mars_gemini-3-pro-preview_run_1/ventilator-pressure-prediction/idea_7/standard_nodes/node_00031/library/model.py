import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale CNN Stem using Inception-like parallel 1D convolutions.
    Captures local features at different temporal resolutions.
    """

    def __init__(self, input_dim, filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            # Padding is calculated to maintain temporal sequence length (same padding)
            self.convs.append(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=filters,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        # Permute to (Batch, Features, Seq_Len) for Conv1d
        x = x.transpose(1, 2)

        outputs = []
        for conv in self.convs:
            out = self.activation(conv(x))
            outputs.append(out)

        # Concatenate along channel dimension
        x_cat = torch.cat(outputs, dim=1)

        # Permute back to (Batch, Seq_Len, Channels)
        x_cat = x_cat.transpose(1, 2)
        return x_cat


class ResidualBlock(nn.Module):
    """
    Bidirectional LSTM block with Projected Residual Connections.
    Cite solution_lesson_node_00027: Use projection shortcuts for dimension-mismatched residuals.
    Cite solution_lesson_node_00030: Avoid redundant pointwise gating.
    """

    def __init__(self, input_dim, hidden_dim, bidirectional, dropout):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Projection layer for residual connection if dimensions mismatch
        if input_dim != self.output_dim:
            self.projection = nn.Linear(input_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # 1. Recurrent Processing
        h, _ = self.lstm(x)  # h: (Batch, Seq_Len, Output_Dim)

        # 2. Projected Residual Connection
        # Cite solution_lesson_node_00024: Apply Dropout only to the residual branch (h), not the identity (x)
        # y = Project(x) + Dropout(LSTM(x))
        res = self.projection(x)
        out = res + self.dropout(h)

        return out


class VentilatorModel(nn.Module):
    """
    Residual Multi-Scale CNN-LSTM Architecture.
    """

    def __init__(self):
        super().__init__()

        # --- Input Dimensions ---
        # Determined by the number of continuous features generated in the pipeline
        input_dim = len(Config.CONT_FEATURES)

        # --- Multi-Scale Stem ---
        self.stem = MultiScaleStem(
            input_dim=input_dim,
            filters=Config.CNN_FILTERS,
            kernel_sizes=Config.CNN_KERNEL_SIZES,
        )

        # Calculate output dimension of the stem (filters * number of kernels)
        stem_out_dim = Config.CNN_FILTERS * len(Config.CNN_KERNEL_SIZES)

        # --- Backbone (Stacked Residual Blocks) ---
        self.layers = nn.ModuleList()
        current_dim = stem_out_dim

        for _ in range(Config.LSTM_LAYERS):
            block = ResidualBlock(
                input_dim=current_dim,
                hidden_dim=Config.LSTM_HIDDEN_DIM,
                bidirectional=Config.LSTM_BIDIRECTIONAL,
                dropout=Config.LSTM_DROPOUT,
            )
            self.layers.append(block)
            # Update current_dim for the next layer (output of BiLSTM is hidden * 2)
            current_dim = Config.LSTM_HIDDEN_DIM * (
                2 if Config.LSTM_BIDIRECTIONAL else 1
            )

        # --- Head ---
        # Projects hidden states to scalar pressure prediction
        self.head = nn.Sequential(
            nn.Linear(current_dim, Config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)

        # Apply Multi-Scale Stem
        x = self.stem(x)

        # Apply Stacked Gated Residual Blocks
        for layer in self.layers:
            x = layer(x)

        # Apply Head to generate predictions
        out = self.head(x)

        return out
