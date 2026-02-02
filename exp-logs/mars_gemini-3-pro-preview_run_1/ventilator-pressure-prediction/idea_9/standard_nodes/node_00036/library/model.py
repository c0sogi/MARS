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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Sequence Data.
    Performs Global Average Pooling over the time dimension to capture
    global context and recalibrate channel weights.
    Cite solution_lesson_node_00030
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Seq, Channels)
        b, s, c = x.size()
        # Permute to (Batch, Channels, Seq) for pooling
        y = x.permute(0, 2, 1)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y


class PhysicsInjectedNet(nn.Module):
    """
    Residual Multi-Scale CNN-LSTM with Squeeze-and-Excitation (SE) Blocks.
    Implements the 'Idea 6' architecture which achieved ~0.31 MAE.
    Includes:
    - Multi-Scale CNN Stem (Cite solution_lesson_node_00015)
    - Deep Residual Bi-LSTM Backbone (Cite solution_lesson_node_00023)
    - Projection Shortcuts for Dimension Mismatch (Cite solution_lesson_node_00027)
    - SE Blocks for Global Context (Cite solution_lesson_node_00030)
    - Dropout on Residual Branch (Cite solution_lesson_node_00024)
    """

    def __init__(self):
        super().__init__()
        self.config = Config

        input_dim = len(self.config.FEATURE_COLS)

        # 1. Stem: Multi-Scale CNN
        self.stem = MultiScaleCNN(
            input_dim=input_dim,
            filters=self.config.CNN_FILTERS,
            kernels=self.config.CNN_KERNELS,
        )
        self.stem_act = nn.GELU()

        # Calculate output dimension of the stem
        stem_out_dim = self.config.CNN_FILTERS * len(self.config.CNN_KERNELS)

        # 2. Backbone: Residual Bi-LSTM with SE
        self.lstm_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = stem_out_dim
        hidden_dim = self.config.LSTM_HIDDEN_DIM
        bidirectional = self.config.BIDIRECTIONAL
        num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * num_directions

        for _ in range(self.config.LSTM_LAYERS):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )

            self.se_blocks.append(SEBlock(lstm_out_dim))

            # Residual Projection
            if current_dim != lstm_out_dim:
                self.projections.append(nn.Linear(current_dim, lstm_out_dim))
            else:
                self.projections.append(nn.Identity())

            self.dropouts.append(nn.Dropout(self.config.DROPOUT))

            current_dim = lstm_out_dim

        # 3. Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Features)

        # --- Stem ---
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.stem_act(h)
        h = h.transpose(1, 2)

        # --- Backbone ---
        for i in range(len(self.lstm_layers)):
            # LSTM
            out, _ = self.lstm_layers[i](h)

            # SE Block
            out = self.se_blocks[i](out)

            # Residual Connection
            # Path A: Projection of input
            res_path = self.projections[i](h)

            # Path B: Dropout on transformed output
            out_path = self.dropouts[i](out)

            # Additive Residual (Cite solution_lesson_node_00033)
            h = res_path + out_path

        # --- Head ---
        preds = self.head(h)
        return preds.squeeze(-1)
