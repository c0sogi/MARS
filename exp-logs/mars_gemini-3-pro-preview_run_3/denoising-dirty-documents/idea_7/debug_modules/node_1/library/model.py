import torch
import torch.nn as nn
from library.config import Config


class DenseLayer(nn.Module):
    """
    Basic unit of the Residual Dense Block.
    Consists of a 3x3 Convolution followed by ReLU.
    """

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class RDB(nn.Module):
    """
    Residual Dense Block (RDB).
    Features dense connectivity where each layer receives inputs from all preceding layers.
    Includes Local Feature Fusion (LFF) and Residual Scaling for stability.
    """

    def __init__(self, in_channels, growth_rate, num_layers, residual_scale):
        super(RDB, self).__init__()
        self.residual_scale = residual_scale
        self.layers = nn.ModuleList()

        # In RDN, the input to layer k is the concatenation of:
        # [block_input, output_layer_1, ..., output_layer_k-1]
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate

        # Local Feature Fusion (LFF): 1x1 Conv to fuse all features back to in_channels
        self.lff = nn.Conv2d(
            current_channels, in_channels, kernel_size=1, padding=0, bias=True
        )

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            # Dense connection: concatenate all previous feature maps
            cat_input = torch.cat(inputs, dim=1)
            out = layer(cat_input)
            inputs.append(out)

        # Fuse all concatenated features
        lff_in = torch.cat(inputs, dim=1)
        lff_out = self.lff(lff_in)

        # Residual Scaling: Scale the residual branch before adding to identity
        return x + lff_out * self.residual_scale


class SRDN(nn.Module):
    """
    Stabilized Residual Dense Network (S-RDN).
    Uses RDBs for dense feature extraction and predicts the noise residual.
    """

    def __init__(self):
        super(SRDN, self).__init__()

        # Load hyperparameters from Config
        in_channels = Config.IN_CHANNELS
        out_channels = Config.OUT_CHANNELS
        num_features = Config.NUM_FEATURES
        growth_rate = Config.GROWTH_RATE
        num_blocks = Config.NUM_RDN_BLOCKS
        num_layers_per_block = Config.NUM_LAYERS_PER_BLOCK
        residual_scale = Config.RESIDUAL_SCALE

        # Shallow Feature Extraction (SFE)
        # SFE1: Extract features from input image
        self.sfe1 = nn.Conv2d(
            in_channels, num_features, kernel_size=3, padding=1, bias=True
        )
        # SFE2: Further processing before entering RDBs
        self.sfe2 = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1, bias=True
        )

        # Stack of Residual Dense Blocks
        self.rdbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdbs.append(
                RDB(num_features, growth_rate, num_layers_per_block, residual_scale)
            )

        # Global Feature Fusion (GFF)
        # Fuses outputs from all RDBs
        self.gff_conv1 = nn.Conv2d(
            num_features * num_blocks, num_features, kernel_size=1, padding=0, bias=True
        )
        self.gff_conv2 = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1, bias=True
        )

        # Final Convolution to project to image space (predicting noise)
        self.final_conv = nn.Conv2d(
            num_features, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        # 1. Shallow Feature Extraction
        f_minus_1 = self.sfe1(x)
        f_0 = self.sfe2(f_minus_1)

        # 2. Residual Dense Blocks
        rdb_outputs = []
        f_curr = f_0
        for rdb in self.rdbs:
            f_curr = rdb(f_curr)
            rdb_outputs.append(f_curr)

        # 3. Global Feature Fusion
        # Concatenate outputs of all RDBs
        gff_in = torch.cat(rdb_outputs, dim=1)
        # 1x1 Conv fusion
        gff_out = self.gff_conv1(gff_in)
        # 3x3 Conv refinement
        gff_out = self.gff_conv2(gff_out)

        # 4. Global Residual Learning (Feature Space)
        # Add the initial shallow features to the fused deep features
        f_df = f_minus_1 + gff_out

        # 5. Predict Noise Residual
        noise = self.final_conv(f_df)

        # 6. Denoising: Clean Image = Input - Noise
        clean = x - noise

        return clean
