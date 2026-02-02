import torch
import torch.nn as nn
from library import config


class DenseLayer(nn.Module):
    """
    Basic unit of the Residual Dense Block.
    Performs Conv -> ReLU and concatenates input with output.
    """

    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(DenseLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        # Dense connection: concatenate input features with new features
        return torch.cat([x, out], 1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    Contains multiple DenseLayers, Local Feature Fusion (LFF), and Local Residual Learning.
    """

    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=3):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        # Add dense layers
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, kernel_size)
            )

        # Local Feature Fusion (LFF)
        # Input to LFF is the concatenation of block input + all dense layer outputs
        lff_in_channels = in_channels + num_layers * growth_rate
        self.lff = nn.Conv2d(lff_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        # Fuse features and project back to original channel size
        out = self.lff(out)

        # Local Residual Learning
        return out + x


class RDN(nn.Module):
    """
    Residual Dense Network (RDN) for Image Denoising.
    Predicts the noise residual of the input image.
    """

    def __init__(
        self,
        channel=config.IMG_CHANNELS,
        growth_rate=config.RDN_GROWTH_RATE,
        num_features=config.RDN_NUM_FEATURES,
        num_blocks=config.RDN_NUM_BLOCKS,
        num_layers=config.RDN_LAYERS_PER_BLOCK,
        kernel_size=config.RDN_KERNEL_SIZE,
    ):
        super(RDN, self).__init__()

        # Shallow Feature Extraction (SFE)
        # SFE1
        padding = (kernel_size - 1) // 2
        self.sfe1 = nn.Conv2d(
            channel, num_features, kernel_size=kernel_size, padding=padding
        )
        # SFE2
        self.sfe2 = nn.Conv2d(
            num_features, num_features, kernel_size=kernel_size, padding=padding
        )

        # Residual Dense Blocks (RDBs)
        self.rdbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdbs.append(
                ResidualDenseBlock(num_features, growth_rate, num_layers, kernel_size)
            )

        # Global Feature Fusion (GFF)
        # Concatenates outputs from all RDBs
        self.gff = nn.Sequential(
            nn.Conv2d(
                num_blocks * num_features, num_features, kernel_size=1, padding=0
            ),
            nn.Conv2d(
                num_features, num_features, kernel_size=kernel_size, padding=padding
            ),
        )

        # Output Layer
        # Maps features to the noise residual (same channels as input)
        self.output = nn.Conv2d(
            num_features, channel, kernel_size=kernel_size, padding=padding
        )

        self._initialize_weights()

    def forward(self, x):
        # Shallow Feature Extraction
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        # RDBs
        x_rdb = sfe2
        rdb_outs = []
        for rdb in self.rdbs:
            x_rdb = rdb(x_rdb)
            rdb_outs.append(x_rdb)

        # Global Feature Fusion
        # Concatenate outputs of all RDBs
        x_cat = torch.cat(rdb_outs, dim=1)
        x_gff = self.gff(x_cat)

        # Global Residual Learning
        # Add SFE1 output to GFF output
        x_gf = sfe1 + x_gff

        # Final prediction (Noise Residual)
        noise_pred = self.output(x_gf)

        return noise_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
