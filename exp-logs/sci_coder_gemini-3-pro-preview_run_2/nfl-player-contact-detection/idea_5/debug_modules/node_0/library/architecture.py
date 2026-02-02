import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with two convolution layers, Batch Norm, and ReLU.
    Maintains the temporal dimension (padding='same' equivalent).
    """

    def __init__(self, channels, kernel_size, dropout=0.0):
        super(ResidualBlock1D, self).__init__()
        # Calculate padding to keep output length equal to input length
        # Assuming stride=1 and dilation=1
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class CFTCN(nn.Module):
    """
    Center-Focused Temporal Convolutional Network (CF-TCN).

    This model processes a flattened wide input vector containing features over a temporal window.
    It uses 1D convolutions to capture temporal patterns while explicitly preserving the
    raw features of the center frame via a skip connection to the classification head.
    """

    def __init__(self):
        super(CFTCN, self).__init__()

        # --- Hyperparameters from Config ---
        self.num_features = Config.NUM_FEATURES_PER_STEP
        self.window_size = Config.WINDOW_SIZE
        self.half_window = Config.HALF_WINDOW_SIZE

        self.cnn_filters = Config.CNN_FILTERS
        self.kernel_size = Config.CNN_KERNEL_SIZE
        self.num_layers = Config.CNN_LAYERS

        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT

        # --- Architecture Components ---

        # 1. Input Projection
        # Maps raw features to the hidden channel dimension of the CNN
        self.input_proj = nn.Conv1d(
            in_channels=self.num_features, out_channels=self.cnn_filters, kernel_size=1
        )

        # 2. Temporal Encoder (Stack of Residual Blocks)
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                ResidualBlock1D(self.cnn_filters, self.kernel_size, dropout=0.1)
            )
        self.encoder = nn.Sequential(*layers)

        # 3. Classification Head
        # The input to the head is the flattened encoder output + the raw center features
        self.encoder_flat_dim = self.cnn_filters * self.window_size
        self.center_dim = self.num_features
        self.head_input_dim = self.encoder_flat_dim + self.center_dim

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),  # Output: Single logit
        )

        # Initialize weights
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
        """
        Args:
            x (torch.Tensor): Flattened input tensor of shape (Batch, INPUT_WIDTH).
                              INPUT_WIDTH = NUM_FEATURES_PER_STEP * WINDOW_SIZE.

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        batch_size = x.size(0)

        # 1. Reshape Input
        # Input x is (Batch, Features * Window).
        # We assume the features are ordered by timestep: [Step0_AllFeats, Step1_AllFeats, ...]
        # Reshape to (Batch, Window, Features)
        x = x.view(batch_size, self.window_size, self.num_features)

        # Permute to (Batch, Features, Window) for Conv1d
        x = x.permute(0, 2, 1)

        # 2. Center-Skip Connection
        # Extract the raw features at the center time step (t=0)
        # Shape: (Batch, Features)
        center_features = x[:, :, self.half_window]

        # 3. Temporal Encoding
        # Project and pass through residual stack
        out = self.input_proj(x)
        out = self.encoder(out)

        # 4. Flatten Encoder Output
        # Shape: (Batch, CNN_FILTERS * WINDOW_SIZE)
        out = out.view(batch_size, -1)

        # 5. Concatenate
        # Combine temporal context with precise center-frame kinematics
        combined = torch.cat([out, center_features], dim=1)

        # 6. Prediction
        logits = self.head(combined)

        return logits
