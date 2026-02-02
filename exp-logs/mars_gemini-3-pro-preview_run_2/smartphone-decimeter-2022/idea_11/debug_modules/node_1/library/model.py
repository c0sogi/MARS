import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and Dropout.
    Structure: Conv1D -> BN -> ReLU -> Dropout -> Conv1D -> BN
    Skip Connection: Input + Output (with projection if dimensions change)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResidualBlock1D, self).__init__()

        # Ensure padding maintains temporal dimension
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class IMULocalTrajectoryCNN(nn.Module):
    """
    Residual 1D-CNN with Multi-Modal Fusion for Trajectory Refinement.

    Architecture:
    1. Input: (Batch, Window_Size, Input_Dim)
    2. Transpose to (Batch, Input_Dim, Window_Size)
    3. Stem: Conv1D -> BN -> ReLU
    4. Backbone: Stack of ResidualBlock1D
    5. Flatten: Preserves temporal context by flattening (Channels * Window_Size)
    6. Head: MLP to predict (DeltaEast, DeltaNorth)
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        window_size=Config.WINDOW_SIZE,
        output_dim=Config.OUTPUT_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_dropout=Config.CNN_DROPOUT,
        mlp_hidden_dims=Config.MLP_HIDDEN_DIMS,
        mlp_dropout=Config.MLP_DROPOUT,
    ):
        super(IMULocalTrajectoryCNN, self).__init__()

        self.window_size = window_size

        # 1. Stem Layer
        # Projects input features to the first channel dimension
        first_channel = cnn_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_dim,
                first_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(first_channel),
            nn.ReLU(inplace=True),
        )

        # 2. Residual Backbone
        layers = []
        in_ch = first_channel

        for out_ch in cnn_channels:
            # Add a residual block for each channel configuration
            # We can stack multiple blocks per stage if needed, but here we do 1 per stage
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel_size, cnn_dropout))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        # 3. Prediction Head (MLP)
        # Flattened size = Final_Channels * Window_Size
        flattened_dim = cnn_channels[-1] * window_size

        mlp_layers = []
        curr_dim = flattened_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(curr_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(mlp_dropout))
            curr_dim = hidden_dim

        # Final output layer
        mlp_layers.append(nn.Linear(curr_dim, output_dim))

        self.head = nn.Sequential(*mlp_layers)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (Batch, Window_Size, Input_Dim)

        # Permute for Conv1D: (Batch, Input_Dim, Window_Size)
        x = x.permute(0, 2, 1)

        # Pass through Stem
        x = self.stem(x)

        # Pass through Backbone
        x = self.backbone(x)

        # Flatten: (Batch, Channels * Window_Size)
        # We do NOT use Global Average Pooling because we want to preserve
        # the specific temporal structure relative to the center epoch.
        x = x.view(x.size(0), -1)

        # Pass through MLP Head
        out = self.head(x)

        return out
