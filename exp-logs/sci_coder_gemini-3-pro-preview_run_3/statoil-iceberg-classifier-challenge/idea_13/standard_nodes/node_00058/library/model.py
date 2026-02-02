import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DSN_CNN(nn.Module):
    """
    Dual-Scale Normalized Simple CNN (DSN-CNN).

    Architecture:
    - 4-Block Backbone (Conv-BN-ReLU-Pool)
    - Selective Hierarchical Pooling (Block 3 & Block 4)
    - Feature Fusion with Normalized Incidence Angle
    - Single Hidden Layer Classification Head
    """

    def __init__(self):
        super(DSN_CNN, self).__init__()

        # Retrieve configuration
        channels = Config.CONV_CHANNELS  # [64, 128, 128, 128]
        fc_dim = Config.FC_DIM
        dropout_rate = Config.DROPOUT_RATE
        input_channels = Config.NUM_CHANNELS

        # --- Backbone ---

        # Block 1: 3 -> 64
        # Captures low-level features, but usually noisy in SAR.
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 64 -> 128
        # Early expansion to preserve texture details.
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 128 -> 128
        # Medium-scale context (approx 9x9 grid). Used for pooling.
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 128 -> 128
        # Abstract existence features (approx 4x4 grid). Used for pooling.
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classification Head ---

        # Input dimension calculation:
        # We use Global Max Pooling on Block 3 and Block 4 outputs.
        # This results in a vector of size [Batch, Channels].
        # We concatenate Block 3 features + Block 4 features + Incidence Angle (1).

        self.dense_input_dim = channels[2] + channels[3] + 1

        self.classifier = nn.Sequential(
            nn.Linear(self.dense_input_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_dim, 1),
        )

        # Note: Weights are initialized using PyTorch defaults (Kaiming Uniform)
        # as per the "Training" section of the Idea description.

    def forward(self, x, angle):
        """
        Forward pass of the DSN-CNN.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, 75, 75).
            angle (torch.Tensor): Incidence angle tensor of shape (B, 1) or (B,).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Ensure angle has correct shape (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Pass through backbone blocks
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        # Selective Hierarchical Pooling (Global Max Pooling)
        # Apply to Block 3 and Block 4 to capture multi-scale peaks

        # x3 shape: (B, 128, H3, W3) -> p3 shape: (B, 128)
        p3 = F.adaptive_max_pool2d(x3, output_size=1).view(x3.size(0), -1)

        # x4 shape: (B, 128, H4, W4) -> p4 shape: (B, 128)
        p4 = F.adaptive_max_pool2d(x4, output_size=1).view(x4.size(0), -1)

        # Feature Fusion
        # Concatenate pooled features and normalized incidence angle
        combined = torch.cat((p3, p4, angle), dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
