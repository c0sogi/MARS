import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions.
    Maintains full resolution (no pooling) while increasing receptive field.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Residual Add -> ReLU
    """

    def __init__(self, channels: int, dilation: int):
        super(DilatedResidualBlock, self).__init__()
        # Padding equals dilation to maintain spatial dimensions for kernel_size=3
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class WSDN_ABS(nn.Module):
    """
    Wide Sequential Dilated Network with Auxiliary Boundary Supervision.

    Features:
    - Learnable 2.5D Projection (65 -> 64 channels)
    - Wide Sequential Backbone (64 channels, no U-Net skips)
    - Dilated Convolutions for large context without downsampling
    - Dual Heads: Primary Ink Mask + Auxiliary Boundary
    """

    def __init__(
        self,
        in_channels: int = Config.Z_DIM,
        model_channels: int = Config.MODEL_CHANNELS,
        dilation_rates: list = Config.DILATION_RATES,
    ):
        super(WSDN_ABS, self).__init__()

        # 1. Learnable 2.5D Projection
        # Projects the volumetric Z-slices into the model's feature space.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Wide Sequential Dilated Backbone
        # Stacks dilated residual blocks sequentially.
        layers = []
        for rate in dilation_rates:
            layers.append(DilatedResidualBlock(model_channels, dilation=rate))

        self.backbone = nn.Sequential(*layers)

        # 3. Output Heads
        # Primary Head: Predicts Ink vs No-Ink
        self.mask_head = nn.Conv2d(model_channels, 1, kernel_size=1)

        # Auxiliary Head: Predicts Ink Boundaries (Edges)
        # Acts as a structural regularizer during training
        self.boundary_head = nn.Conv2d(model_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor of shape (B, Z_DIM, H, W)

        Returns:
            Dictionary containing:
                'mask': Logits for ink segmentation (B, 1, H, W)
                'boundary': Logits for boundary detection (B, 1, H, W)
        """
        # Project input volume to feature space
        x = self.projection(x)

        # Extract features using dilated backbone
        features = self.backbone(x)

        # Generate predictions
        mask_logits = self.mask_head(features)
        boundary_logits = self.boundary_head(features)

        return {"mask": mask_logits, "boundary": boundary_logits}
