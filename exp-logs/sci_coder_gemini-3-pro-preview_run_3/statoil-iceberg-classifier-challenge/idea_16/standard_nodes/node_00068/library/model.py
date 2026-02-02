import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-Layer CNN Architecture.
    Based on the 'Current Best' solution strategy:
    - 4 Conv Blocks with MaxPooling and Dropout.
    - Global Max Pooling at the end (no hierarchical pooling).
    - Late Fusion with raw incidence angle.
    - Single hidden layer in classification head.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Hyperparameters
        channels = Config.CHANNEL_SIZES  # Expected: [64, 128, 128, 128]
        spatial_dropout = Config.SPATIAL_DROPOUT_RATE
        head_dropout = Config.HEAD_DROPOUT_RATE

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS, channels[0], kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=spatial_dropout),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=spatial_dropout),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=spatial_dropout),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=spatial_dropout),
        )

        # Global Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Input: 128 (CNN features) + 1 (Angle)
        self.head = nn.Sequential(
            nn.Linear(channels[3] + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform / Fan-In).
        Cite solution_lesson_node_00045, solution_lesson_node_00064.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, inc_angle):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling
        # Cite solution_lesson_node_00007, solution_lesson_node_00034
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Late Fusion
        # Cite solution_lesson_node_00039, solution_lesson_node_00057
        angle = inc_angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        logits = self.head(x)
        return logits
