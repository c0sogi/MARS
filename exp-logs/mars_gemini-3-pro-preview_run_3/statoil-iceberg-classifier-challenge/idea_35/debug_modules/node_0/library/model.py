import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Module.
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the intermediate FC layer.
        """
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DualPolarityPooling(nn.Module):
    def __init__(self):
        """
        Dual-Polarity Pooling Layer.
        Captures both the strongest signal (Max) and the deepest shadow (Min).
        """
        super(DualPolarityPooling, self).__init__()

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Flatten spatial dims: (Batch, Channels, H*W)
        flat = x.view(x.size(0), x.size(1), -1)

        # Global Max Pooling (Signal Peaks)
        max_val, _ = torch.max(flat, dim=2)  # (Batch, Channels)

        # Global Min Pooling (Signal Voids/Shadows)
        min_val, _ = torch.min(flat, dim=2)  # (Batch, Channels)

        # Concatenate: (Batch, 2 * Channels)
        return torch.cat([max_val, min_val], dim=1)


class DPACNN(nn.Module):
    def __init__(self):
        """
        Dual-Polarity Attentive CNN (DPA-CNN).
        Custom 4-Stage Attentive CNN optimized for SAR imagery.
        """
        super(DPACNN, self).__init__()

        # Configuration
        # Input: 3 channels (HH, HV, Avg)
        # Block 1: 64 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(64, reduction=16),
            nn.MaxPool2d(2),  # 75 -> 37
        )

        # Block 2: 128 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128, reduction=16),
            nn.MaxPool2d(2),  # 37 -> 18
        )

        # Block 3: 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128, reduction=16),
            nn.MaxPool2d(2),  # 18 -> 9
        )

        # Block 4: 128 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128, reduction=16),
            nn.MaxPool2d(2),  # 9 -> 4
        )

        # Readout
        self.pooling = DualPolarityPooling()

        # Classifier
        # Input features: 128 (Max) + 128 (Min) = 256
        # Plus 1 scalar for incidence angle = 257
        self.classifier = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Applies Kaiming Uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x: Input images (Batch, 3, 75, 75)
            angle: Incidence angles (Batch,) or (Batch, 1)
        Returns:
            Logits (Batch, 1)
        """
        # Backbone
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (B, 128, 4, 4)

        # Dual-Polarity Pooling
        x = self.pooling(x)  # (B, 256)

        # Feature Fusion
        angle = angle.view(-1, 1)  # Ensure (B, 1)
        x = torch.cat([x, angle], dim=1)  # (B, 257)

        # Classification
        x = self.classifier(x)

        return x
