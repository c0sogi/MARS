import torch
import torch.nn as nn
from library.config import Config


class ZeroGammaResBlock(nn.Module):
    """
    A Residual Block with Zero-Gamma Initialization.
    Structure: Conv -> BN -> ReLU -> Conv -> BN

    The second Batch Normalization layer's gamma (weight) is initialized to 0.
    This ensures the block behaves as an identity mapping at the start of training,
    facilitating the training of very deep networks.
    """

    def __init__(self, channels):
        super(ZeroGammaResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize first Conv layer
        nn.init.kaiming_normal_(
            self.layers[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # Initialize first BN layer (standard)
        nn.init.constant_(self.layers[1].weight, 1)
        nn.init.constant_(self.layers[1].bias, 0)

        # Initialize second Conv layer
        nn.init.kaiming_normal_(
            self.layers[3].weight, mode="fan_out", nonlinearity="relu"
        )

        # Initialize second BN layer (Zero-Gamma)
        # Setting weight to 0 makes the output of this block 0 initially (assuming bias is 0)
        # So the residual connection x + block(x) becomes x + 0 = x (Identity)
        nn.init.constant_(self.layers[4].weight, 0)
        nn.init.constant_(self.layers[4].bias, 0)

    def forward(self, x):
        return x + self.layers(x)


class ResDnCNN(nn.Module):
    """
    High-Capacity Zero-Initialized Deep Residual Ensemble (HC-ZI-ResDnCNN).

    This network predicts the noise residual of an input image.
    It uses a deep stack of ZeroGammaResBlocks without pooling to preserve spatial resolution.
    """

    def __init__(
        self, depth=None, filters=None, input_channels=None, output_channels=None
    ):
        super(ResDnCNN, self).__init__()

        # Load hyperparameters from Config if not provided
        self.depth = depth if depth is not None else Config.MODEL_DEPTH
        self.filters = filters if filters is not None else Config.MODEL_FILTERS
        self.input_channels = (
            input_channels if input_channels is not None else Config.INPUT_CHANNELS
        )
        self.output_channels = (
            output_channels if output_channels is not None else Config.OUTPUT_CHANNELS
        )

        # Head: Feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(
                self.input_channels, self.filters, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        # Body: Deep stack of residual blocks
        self.body = nn.Sequential(
            *[ZeroGammaResBlock(self.filters) for _ in range(self.depth)]
        )

        # Tail: Reconstruction of the noise residual
        self.tail = nn.Conv2d(
            self.filters, self.output_channels, kernel_size=3, padding=1, bias=True
        )

        self._init_head_tail()

    def _init_head_tail(self):
        # Initialize Head Conv
        nn.init.kaiming_normal_(
            self.head[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if self.head[0].bias is not None:
            nn.init.constant_(self.head[0].bias, 0)

        # Initialize Tail Conv
        # Using fan_out/linear because this is the final regression layer
        nn.init.kaiming_normal_(self.tail.weight, mode="fan_out", nonlinearity="linear")
        if self.tail.bias is not None:
            nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Noisy input image tensor [B, C, H, W].

        Returns:
            torch.Tensor: Predicted noise residual [B, C, H, W].
        """
        feat = self.head(x)
        res = self.body(feat)
        noise = self.tail(res)
        return noise
