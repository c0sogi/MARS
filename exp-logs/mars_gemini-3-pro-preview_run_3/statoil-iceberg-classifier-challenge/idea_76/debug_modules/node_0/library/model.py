import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SqueezeExcitation(nn.Module):
    """
    Standard Squeeze-and-Excitation module.
    Squeezes spatially using Global Average Pooling.
    Excites channels using a bottleneck MLP with ReLU.
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, channels // reduction)
        # 1x1 convolutions act as fully connected layers on the channel dimension
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = x.mean(dim=(2, 3), keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv -> BN -> LeakyReLU -> SE -> MaxPool.
    Retains bias in Conv2d to preserve initialization dynamics.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = SqueezeExcitation(out_channels, reduction=Config.SE_REDUCTION_RATIO)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class ATSICNN(nn.Module):
    """
    Asymmetric Texture-Shadow Isomorphic CNN (ATSI-CNN).

    Architecture:
    - 4-Stage Plain CNN Backbone.
    - Asymmetric Isomorphic Readout from Stage 3 and Stage 4.
    - Stage 3 (Texture Focus): Global Max + Global MAD Pooling.
    - Stage 4 (Shadow Focus): Global Max + Global Min Pooling.
    - Feature Fusion with Raw Incidence Angle.
    """

    def __init__(self):
        super(ATSICNN, self).__init__()

        # --- Backbone ---
        # Stage 1: 3 -> 64 (Output ~37x37)
        self.stage1 = ConvBlock(Config.NUM_CHANNELS, Config.BACKBONE_CHANNELS[0])
        # Stage 2: 64 -> 128 (Output ~18x18)
        self.stage2 = ConvBlock(
            Config.BACKBONE_CHANNELS[0], Config.BACKBONE_CHANNELS[1]
        )
        # Stage 3: 128 -> 128 (Output ~9x9)
        self.stage3 = ConvBlock(
            Config.BACKBONE_CHANNELS[1], Config.BACKBONE_CHANNELS[2]
        )
        # Stage 4: 128 -> 128 (Output ~4x4)
        self.stage4 = ConvBlock(
            Config.BACKBONE_CHANNELS[2], Config.BACKBONE_CHANNELS[3]
        )

        # --- Asymmetric Projections ---
        # Decoupled 1x1 Convs to reduce dimensions before pooling
        self.proj_s3 = nn.Conv2d(
            Config.BACKBONE_CHANNELS[2], Config.PROJECTION_DIM, kernel_size=1, bias=True
        )
        self.proj_s4 = nn.Conv2d(
            Config.BACKBONE_CHANNELS[3], Config.PROJECTION_DIM, kernel_size=1, bias=True
        )

        # --- Classification Head ---
        # Input: Image Features (256) + Angle (1)
        self.fc1 = nn.Linear(Config.CLASSIFIER_INPUT_DIM, Config.CLASSIFIER_HIDDEN_DIM)
        self.act_fc = nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True)
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(Config.CLASSIFIER_HIDDEN_DIM, Config.NUM_CLASSES)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform initialization for weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=Config.LEAKY_RELU_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=Config.LEAKY_RELU_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Image input (B, 3, 75, 75).
            angle (torch.Tensor): Incidence angle input (B,).
        """
        # Backbone Forward
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # Stage 3 Output
        x4 = self.stage4(x3)  # Stage 4 Output

        # --- Stage 3 Processing (Texture Focus) ---
        p3 = self.proj_s3(x3)  # (B, 64, 9, 9)

        # 1. Global Max Pooling (Peak Intensity)
        f3_max = F.adaptive_max_pool2d(p3, (1, 1)).view(p3.size(0), -1)

        # 2. Global MAD Pooling (Texture/Roughness)
        # Mean Absolute Deviation: mean(|x - mean(x)|)
        mu3 = p3.mean(dim=(2, 3), keepdim=True)
        f3_mad = (p3 - mu3).abs().mean(dim=(2, 3))  # Result is (B, 64)

        # --- Stage 4 Processing (Shadow Focus) ---
        p4 = self.proj_s4(x4)  # (B, 64, 4, 4)

        # 3. Global Max Pooling (Peak Intensity)
        f4_max = F.adaptive_max_pool2d(p4, (1, 1)).view(p4.size(0), -1)

        # 4. Global Min Pooling (Shadow Depth)
        # Min(x) = -Max(-x)
        f4_min = -F.adaptive_max_pool2d(-p4, (1, 1)).view(p4.size(0), -1)

        # --- Feature Fusion ---
        # Concatenate all image features: 64*4 = 256
        img_features = torch.cat([f3_max, f3_mad, f4_max, f4_min], dim=1)

        # Concatenate Incidence Angle
        # Angle is (B,), view as (B, 1)
        angle_feat = angle.view(-1, 1)

        combined = torch.cat([img_features, angle_feat], dim=1)

        # --- Classifier ---
        y = self.fc1(combined)
        y = self.act_fc(y)
        y = self.dropout(y)
        y = self.fc2(y)

        return y
