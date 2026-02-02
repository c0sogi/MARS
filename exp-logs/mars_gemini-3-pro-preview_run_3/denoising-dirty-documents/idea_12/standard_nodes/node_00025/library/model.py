import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import N_CHANNELS, N_FEATS, N_RES_BLOCKS, KERNEL_SIZE, REDUCTION


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module.
    Factorizes channel attention into vertical and horizontal directions to preserve
    positional information, optimized for structural data like text.
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool along spatial dimensions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Shared 1x1 Conv -> BN -> Non-linearity
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into vertical and horizontal tensors
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ResidualBlock(nn.Module):
    """
    Residual Block with Coordinate Attention and Zero-Gamma Initialization.
    Structure: Conv-BN-ReLU-Conv-BN-CA + Residual
    """

    def __init__(self, n_feats, kernel_size, reduction=16):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_feats)

        self.ca = CoordAtt(n_feats, reduction)

        # Zero-Gamma Initialization for the last BN in the block
        # This initializes the block as an identity function, stabilizing deep training.
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out)

        out += residual
        return out


class CAResDnCNN(nn.Module):
    """
    Coordinate Attention Stabilized Deep Residual Network (CA-ResDnCNN).
    Predicts the noise residual of the input image.
    """

    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_feats=N_FEATS,
        n_res_blocks=N_RES_BLOCKS,
        kernel_size=KERNEL_SIZE,
        reduction=REDUCTION,
    ):
        super(CAResDnCNN, self).__init__()

        # Head: Input -> Features
        self.head = nn.Conv2d(
            n_channels, n_feats, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # Body: Deep Stack of Residual Blocks
        blocks = []
        for _ in range(n_res_blocks):
            blocks.append(ResidualBlock(n_feats, kernel_size, reduction))
        self.body = nn.Sequential(*blocks)

        # Tail: Features -> Noise Output
        self.tail = nn.Conv2d(
            n_feats, n_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # BN weights are handled in ResidualBlock for the zero-gamma one.
            # Default initialization is used for other BNs (weight=1, bias=0).

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input noisy image (B, C, H, W).
        Returns:
            torch.Tensor: Predicted noise residual (B, C, H, W).
        """
        out = self.head(x)
        out = self.body(out)
        noise = self.tail(out)
        return noise
