import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
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


class DilatedDualBranchBlock(nn.Module):
    """
    A building block that captures multi-scale features using parallel dilated convolutions.
    Structure:
        - Branch 1: 3x3 Conv, Dilation 1 (Local details)
        - Branch 2: 3x3 Conv, Dilation 2 (Context/Shadows)
        - Fusion: Concatenation -> 1x1 Conv
        - Activation: BatchNorm -> LeakyReLU -> SEBlock
        - Downsampling: MaxPool2d
    """

    def __init__(self, in_channels, out_channels):
        super(DilatedDualBranchBlock, self).__init__()

        # Branch 1: Local features (Speckle)
        self.branch1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=True
        )

        # Branch 2: Context features (Shadows/Shape)
        self.branch2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=True
        )

        # Fusion layer to restore target channel width
        self.fusion = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1, bias=True
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = SEBlock(out_channels, reduction=16)

        # Aggressive spatial downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        # Concatenate along channel dimension
        out = torch.cat([b1, b2], dim=1)

        # Fuse and refine
        out = self.fusion(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.se(out)

        # Downsample
        return self.pool(out)


class MSD_SE_CNN(nn.Module):
    """
    Multi-Scale Dilated SE-CNN.
    A 4-stage convolutional network designed for SAR image classification.
    Features:
        - 4-Stage Dilated Dual-Branch Backbone
        - Selective Hierarchical Pooling (Stage 3 + Stage 4)
        - Raw Incidence Angle Injection
        - Single Hidden Layer Classification Head
    """

    def __init__(self):
        super(MSD_SE_CNN, self).__init__()

        # Backbone: 4 Stages
        # Input: 3 channels (HH, HV, Avg)
        self.stage1 = DilatedDualBranchBlock(3, 64)
        self.stage2 = DilatedDualBranchBlock(64, 128)
        self.stage3 = DilatedDualBranchBlock(128, 128)
        self.stage4 = DilatedDualBranchBlock(128, 128)

        # Classification Head
        # Input Dimension:
        #   128 (Stage 3 Global Max Pool)
        # + 128 (Stage 4 Global Max Pool)
        # + 1   (Incidence Angle)
        # = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization (Fan-In) for stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle tensor of shape (B, 1) or (B,)
        Returns:
            logits (torch.Tensor): Raw output scores of shape (B, 1)
        """
        # Forward pass through backbone
        x1 = self.stage1(x)  # -> (B, 64, 37, 37)
        x2 = self.stage2(x1)  # -> (B, 128, 18, 18)
        x3 = self.stage3(x2)  # -> (B, 128, 9, 9)
        x4 = self.stage4(x3)  # -> (B, 128, 4, 4)

        # Selective Hierarchical Pooling
        # Global Max Pooling on Stage 3 (Medium scale features)
        p3 = F.adaptive_max_pool2d(x3, 1).view(x3.size(0), -1)  # (B, 128)

        # Global Max Pooling on Stage 4 (Abstract features)
        p4 = F.adaptive_max_pool2d(x4, 1).view(x4.size(0), -1)  # (B, 128)

        # Ensure angle has correct shape (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Feature Fusion
        # Concatenate pooled features and raw incidence angle
        fused_features = torch.cat([p3, p4, angle], dim=1)  # (B, 257)

        # Classification
        logits = self.head(fused_features)

        return logits
