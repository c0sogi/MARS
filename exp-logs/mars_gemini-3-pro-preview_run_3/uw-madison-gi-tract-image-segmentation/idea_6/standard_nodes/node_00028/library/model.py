import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBnRelu(nn.Module):
    """
    Standard Convolution -> BatchNorm -> ReLU block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    U-Net++ Decoder Block.
    Concatenates inputs from the same level (dense skips) and the upsampled input from the lower level.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # in_channels: channels from the lower level (after upsampling)
        # skip_channels: sum of channels from all skip connections at this level

        self.conv1 = ConvBnRelu(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = ConvBnRelu(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skips):
        # x: input from lower level
        # skips: list of tensors from the same level

        # Upsample x to match skip size
        # We use interpolation for flexibility regarding input sizes
        target_size = skips[0].shape[2:]
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)

        # Concatenate
        # skips is a list [x_{i,0}, x_{i,1}, ...]
        # x is up(x_{i+1, j-1})
        combined = torch.cat(skips + [x], dim=1)

        out = self.conv1(combined)
        out = self.conv2(out)
        return out


class UnetPlusPlus(nn.Module):
    def __init__(self, backbone, in_channels, num_classes, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder
        # Using timm to load EfficientNet-B4 features
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=True, in_chans=in_channels
        )

        # Determine encoder channel counts dynamically
        # EfficientNet-B4 typically: [24, 32, 56, 160, 448] for indices 0-4
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 256, 256)
            enc_features = self.encoder(dummy)
            self.enc_channels = [f.shape[1] for f in enc_features]

        # 2. Decoder Configuration
        # We define decoder channels for levels 0, 1, 2, 3, 4
        # Level 0 is the highest resolution (stride 2 for EffNet)
        # We decrease channels as we go up the U-shape
        self.dec_channels = [16, 32, 64, 128, 256]  # [d0, d1, d2, d3, d4]

        # Ensure we have enough levels from encoder
        if len(self.enc_channels) < 5:
            raise ValueError(
                f"Backbone {backbone} provides fewer than 5 feature levels."
            )

        # 3. Construct Nodes
        # We use a ModuleDict to hold the blocks: 'x_{i}_{j}'
        # i: vertical index (downsampling level), 0 to 4
        # j: horizontal index (dense block), 1 to 4-i
        self.blocks = nn.ModuleDict()

        # Loop over columns (j)
        for j in range(1, 5):
            # Loop over rows (i)
            for i in range(5 - j):
                # Determine input channels from below (upsampled)
                # Input comes from x_{i+1, j-1}
                # If j=1, input is encoder feature x_{i+1, 0}
                if j == 1:
                    in_ch_below = self.enc_channels[i + 1]
                else:
                    in_ch_below = self.dec_channels[i + 1]

                # Determine skip channels from same level
                # Inputs: x_{i, 0}, ..., x_{i, j-1}
                skip_ch = 0
                # Add x_{i,0} (Encoder feature)
                skip_ch += self.enc_channels[i]
                # Add x_{i,1} ... x_{i, j-1} (Decoder features)
                skip_ch += (j - 1) * self.dec_channels[i]

                # Create block
                block_name = f"x_{i}_{j}"
                self.blocks[block_name] = DecoderBlock(
                    in_channels=in_ch_below,
                    skip_channels=skip_ch,
                    out_channels=self.dec_channels[i],
                )

        # 4. Segmentation Heads
        # Attached to x_{0,1}, x_{0,2}, x_{0,3}, x_{0,4}
        # These are all at stride 2 relative to input
        self.seg_heads = nn.ModuleDict()
        if self.deep_supervision:
            for j in range(1, 5):
                self.seg_heads[f"head_{j}"] = nn.Conv2d(
                    self.dec_channels[0], num_classes, kernel_size=1
                )
        else:
            self.seg_heads["head_4"] = nn.Conv2d(
                self.dec_channels[0], num_classes, kernel_size=1
            )

        # Normalization (ImageNet stats)
        # Input is [0, 1], we need to normalize to mean/std
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x is [0, 1] from data loader
        # Normalize to ImageNet stats
        x = (x - self.mean) / self.std

        input_size = x.shape[2:]

        # Encoder Pass
        features = self.encoder(x)
        # features list: [x_0_0, x_1_0, x_2_0, x_3_0, x_4_0]

        # Store nodes in a dict for easy access
        nodes = {}
        for i, f in enumerate(features):
            nodes[f"x_{i}_0"] = f

        # Decoder Grid Pass
        # Compute columns 1 to 4
        for j in range(1, 5):
            for i in range(5 - j):
                # Gather skips: x_{i,0} ... x_{i, j-1}
                skips = [nodes[f"x_{i}_{k}"] for k in range(j)]

                # Get input from below: x_{i+1, j-1}
                x_below = nodes[f"x_{i+1}_{j-1}"]

                # Compute Block
                block = self.blocks[f"x_{i}_{j}"]
                out = block(x_below, skips)

                # Store Result
                nodes[f"x_{i}_{j}"] = out

        # Segmentation Heads
        outputs = []

        # Helper to interpolate logits to original input size
        def process_head(node_name):
            head_idx = node_name.split("_")[-1]
            logits = self.seg_heads[f"head_{head_idx}"](nodes[node_name])
            # Upsample from Stride 2 to Stride 1
            return F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=True
            )

        if self.deep_supervision:
            # Return list: [head(x_0_1), head(x_0_2), head(x_0_3), head(x_0_4)]
            # We return them in order. The loss function will average them.
            outputs.append(process_head("x_0_1"))
            outputs.append(process_head("x_0_2"))
            outputs.append(process_head("x_0_3"))
            outputs.append(process_head("x_0_4"))
        else:
            # Only final output
            outputs = process_head("x_0_4")

        return outputs


def build_model(config):
    """
    Instantiates the U-Net++ model based on the provided configuration.
    """
    model = UnetPlusPlus(
        backbone=config.BACKBONE,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        deep_supervision=config.DEEP_SUPERVISION,
    )
    return model
