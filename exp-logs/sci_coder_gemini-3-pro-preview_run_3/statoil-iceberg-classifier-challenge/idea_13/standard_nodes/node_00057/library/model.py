import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple CNN with Global Max Pooling.
    Optimized based on Lesson solution_lesson_node_00035:
    Prefers hierarchical filtering over multi-scale aggregation for noisy datasets.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Retrieve configuration
        channels = Config.CONV_CHANNELS  # [64, 128, 128, 128]
        fc_dim = Config.FC_DIM
        dropout_rate = Config.DROPOUT_RATE
        input_channels = Config.NUM_CHANNELS

        # --- Backbone ---

        # Block 1: 3 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 128 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 128 -> 128
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classification Head ---

        # Input dimension calculation:
        # We use Global Max Pooling ONLY on Block 4 output.
        # We concatenate Block 4 features + Incidence Angle (1).
        self.dense_input_dim = channels[3] + 1

        self.classifier = nn.Sequential(
            nn.Linear(self.dense_input_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x, angle):
        """
        Forward pass of the SimpleCNN.
        """
        # Ensure angle has correct shape (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Pass through backbone blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling on final block only
        p4 = F.adaptive_max_pool2d(x, output_size=1).view(x.size(0), -1)

        # Feature Fusion
        # Concatenate pooled features and normalized incidence angle
        combined = torch.cat((p4, angle), dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
