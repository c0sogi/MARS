import torch
import torch.nn as nn
from library.config import Config


class ParallelAtrousBlock(nn.Module):
    """
    A block containing parallel convolutional branches with different dilation rates.
    The outputs are fused via concatenation and projection, maintaining full spatial resolution.
    """

    def __init__(self, channels, rates, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList()

        # Create parallel branches for each dilation rate
        for rate in rates:
            # To maintain spatial dimensions with kernel_size=3: padding = dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Fusion layer: Project concatenated features back to original channel width
        # Input channels = channels * number of branches
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(rates), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        # Apply all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along the channel dimension
        concatenated = torch.cat(branch_outputs, dim=1)

        # Project back to original channel size
        out = self.project(concatenated)

        # Residual connection
        return x + out


class PSDN(nn.Module):
    """
    Parallel-Scale Dilated Network (PSDN).

    Architecture:
    1. Learnable 2.5D Projection (Input Z-slices -> Model Channels)
    2. Deep stack of ParallelAtrousBlocks to capture multi-scale context
    3. Final classification head (Model Channels -> 1)
    """

    def __init__(self, num_blocks=6):
        super().__init__()

        # 1. Learnable 2.5D Projection
        # Compresses the 65 input Z-slices into the compact feature space
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.MODEL_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.MODEL_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Backbone: Stack of Parallel Atrous Blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ParallelAtrousBlock(
                    channels=Config.MODEL_CHANNELS,
                    rates=Config.DILATION_RATES,
                    dropout=Config.DROPOUT,
                )
            )
        self.backbone = nn.Sequential(*layers)

        # 3. Classification Head
        self.classifier = nn.Conv2d(Config.MODEL_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 65, H, W)
        Returns:
            logits: Output tensor of shape (Batch, 1, H, W)
        """
        # Project Z-dimension (Batch, 65, H, W) -> (Batch, C, H, W)
        x = self.projection(x)

        # Apply deep parallel dilated backbone
        x = self.backbone(x)

        # Pixel-wise classification
        logits = self.classifier(x)

        return logits
