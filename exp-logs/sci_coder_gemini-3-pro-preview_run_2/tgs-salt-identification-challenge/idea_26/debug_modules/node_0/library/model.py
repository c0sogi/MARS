import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Follows the structure: 1x1 Conv -> 3x3 Transposed Conv -> 1x1 Conv.
    The internal dimension is set to in_channels // 4 to maintain width while
    reducing parameter count, characteristic of the 'Wide' variant logic.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Wide-LinkNet logic: internal dimension is derived from input
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Compress/Project
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv: Upsample
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv: Expand/Project to output
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    Corrected Multi-Task Wide-LinkNet with ResNet34 Backbone.

    Features:
    1. 1-Channel Input Adaptation (Summed Weights).
    2. Auxiliary Depth Regression Head at Bottleneck (No Injection).
    3. Wide-LinkNet Decoder with Additive Skip Connections.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        # Using pretrained weights for faster convergence
        resnet = models.resnet34(pretrained=Config.PRETRAINED)

        # 2. Input Adaptation
        # Modify first layer to accept 1 channel (Config.IN_CHANNELS) instead of 3.
        # We sum the pretrained weights across the channel dimension to preserve
        # the learned edge/texture filters.
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 ch, 1/4 res
        self.layer2 = resnet.layer2  # 128 ch, 1/8 res
        self.layer3 = resnet.layer3  # 256 ch, 1/16 res
        self.layer4 = resnet.layer4  # 512 ch, 1/32 res (Bottleneck)

        # 3. Auxiliary Depth Head
        # Attached to the bottleneck features (Layer 4).
        # Predicts a scalar depth value. This forces the encoder to retain
        # depth-correlated features without injecting depth into the decoder.
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

        # 4. Decoder (Wide-LinkNet)
        # Uses additive skip connections.

        # Dec4: Takes Layer4 (512), Upsamples to 1/16, Adds Layer3 (256)
        # Output channels set to 256 to match Layer3 for addition.
        self.dec4 = DecoderBlock(512, 256)

        # Dec3: Takes Dec4 (256), Upsamples to 1/8, Adds Layer2 (128)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: Takes Dec3 (128), Upsamples to 1/4, Adds Layer1 (64)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: Takes Dec2 (64), Upsamples to 1/2 (64x64), Adds Conv1/Relu (64)
        # Note: Layer1 input was 32x32 (after maxpool). Conv1 output is 64x64.
        # We need to get back to 64x64 here.
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsampling Block
        # Upsamples from 1/2 (64x64) to Full Resolution (128x128)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final Projection to Binary Mask Logits
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.conv1(x)  # 64x64, 64ch
        x0_bn = self.bn1(x0)
        x0_relu = self.relu(x0_bn)

        x1_in = self.maxpool(x0_relu)  # 32x32, 64ch

        e1 = self.layer1(x1_in)  # 32x32, 64ch
        e2 = self.layer2(e1)  # 16x16, 128ch
        e3 = self.layer3(e2)  # 8x8, 256ch
        e4 = self.layer4(e3)  # 4x4, 512ch (Bottleneck)

        # --- Auxiliary Task ---
        # Predict depth from the bottleneck features
        pred_depth = self.depth_head(e4)

        # --- Decoder ---
        # LinkNet style: Decoder Block Output + Encoder Feature (Additive Skip)

        # Block 4
        d4 = self.dec4(e4)
        d4 = d4 + e3

        # Block 3
        d3 = self.dec3(d4)
        d3 = d3 + e2

        # Block 2
        d2 = self.dec2(d3)
        d2 = d2 + e1

        # Block 1
        # Upsample d2 (32x32) -> 64x64. Add x0_relu (64x64).
        d1 = self.dec1(d2)
        d1 = d1 + x0_relu

        # --- Final ---
        # Upsample to 128x128
        out = self.final_up(d1)

        # Project to logits
        logits = self.final_conv(out)

        # Return dictionary for MultiTaskLoss
        return {"mask": logits, "depth": pred_depth}
