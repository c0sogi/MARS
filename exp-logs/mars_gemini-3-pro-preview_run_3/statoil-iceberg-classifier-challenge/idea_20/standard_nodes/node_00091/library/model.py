import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    SimpleCNN Architecture.

    Key Features:
    - 4-Stage Plain CNN Backbone.
    - Early Channel Expansion (64 -> 128 -> 128 -> 128).
    - Global Max Pooling (Cite Lesson 007).
    - LeakyReLU Activation (Cite Lesson 078).
    - Late Fusion with Raw Incidence Angle (Cite Lesson 039).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Hyperparameters from Config
        in_channels = Config.INPUT_CHANNELS
        widths = Config.CHANNEL_WIDTHS  # Expected: [64, 128, 128, 128]
        dropout_p = Config.FC_DROPOUT

        # Activation: LeakyReLU to prevent dead neurons and align with default init
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Block 1: Input -> 64
        self.conv1 = nn.Conv2d(
            in_channels, widths[0], kernel_size=3, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64 -> 128
        self.conv2 = nn.Conv2d(
            widths[0], widths[1], kernel_size=3, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(widths[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(
            widths[1], widths[2], kernel_size=3, padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(widths[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(
            widths[2], widths[3], kernel_size=3, padding=1, bias=True
        )
        self.bn4 = nn.BatchNorm2d(widths[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification Head
        # Fusion: Global Max Pool (128) + Angle (1)
        fusion_dim = widths[3] + 1
        hidden_dim = 512

        self.head_fc = nn.Linear(fusion_dim, hidden_dim)
        self.head_dropout = nn.Dropout(p=dropout_p)
        self.head_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, angle):
        # Stage 1
        x = self.pool1(self.act(self.bn1(self.conv1(x))))
        # Stage 2
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        # Stage 3
        x = self.pool3(self.act(self.bn3(self.conv3(x))))
        # Stage 4
        x = self.pool4(self.act(self.bn4(self.conv4(x))))

        # Global Max Pooling
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Fusion
        angle = angle.view(-1, 1)
        fused = torch.cat([x, angle], dim=1)

        # Classifier
        x = self.act(self.head_fc(fused))
        x = self.head_dropout(x)
        logits = self.head_out(x)

        return logits.squeeze(1)


# Alias for backward compatibility and refactoring support
SelectiveSECNN = SimpleCNN
