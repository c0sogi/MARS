import torch
import torch.nn as nn
import torch.nn.init as init
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Linear Residual Stack (Conv-BN-ReLU-Conv-BN) with Zero-Gamma Initialization.
    This block is designed to be identity-initialized to facilitate training of very deep networks.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        # First layer of the block
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Second layer of the block
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for convolutions
        init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        # Standard initialization for the first BN
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)

        # Zero-Gamma initialization for the second BN
        # This makes the block act as an identity function at initialization
        init.constant_(self.bn2.weight, 0)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Add residual to input (Identity Shortcut)
        return x + residual


class DnCNN(nn.Module):
    """
    ZI-ResDnCNN: Zero-Initialized Deep Residual Denoising Network.

    Architecture:
        Input -> Head (Conv+ReLU) -> Deep Body (Stack of ResidualBlocks) -> Tail (Conv) -> Output (Noise)

    The network predicts the noise residual R(x). The clean image is obtained via Input - R(x).
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_RES_BLOCKS,
    ):
        super(DnCNN, self).__init__()

        # Head: Initial feature extraction
        # Bias is True because there is no BN immediately following
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Body: Deep stack of residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # Tail: Reconstruction of the noise residual
        # Bias is True because there is no BN immediately following
        self.tail = nn.Conv2d(
            num_features, out_channels, kernel_size=3, padding=1, bias=True
        )

        self._init_head_tail()

    def _init_head_tail(self):
        # Initialize Head
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Initialize Tail
        # Using Kaiming Normal assuming linear activation (effectively) for the output
        init.kaiming_normal_(self.tail.weight, mode="fan_out", nonlinearity="linear")
        if self.tail.bias is not None:
            init.constant_(self.tail.bias, 0)

    def forward(self, x):
        """
        Forward pass to predict the noise.

        Args:
            x (torch.Tensor): Input noisy image tensor [B, C, H, W].

        Returns:
            torch.Tensor: Predicted noise tensor [B, C, H, W].
        """
        out = self.head(x)
        out = self.body(out)
        noise = self.tail(out)
        return noise
