import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.

    Structure:
    1. 1x1 Conv (Reduce channels to internal width)
    2. Transposed Conv (Upsample)
    3. 1x1 Conv (Expand channels to out_channels)
    4. Add Skip Connection (if provided)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Internal width logic: Standard LinkNet reduces to in_channels // 4.
        # We ensure a minimum width (32) to preserve information flow.
        mid_channels = max(in_channels // 4, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.trans = nn.Sequential(
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.trans(x)
        x = self.conv2(x)
        if skip is not None:
            # Additive skip connection (LinkNet style)
            x = x + skip
        return x


class SaltModel(nn.Module):
    """
    Salt Segmentation Model based on ResNet34 Encoder and Wide-LinkNet Decoder.

    Modes:
    - 'teacher': Includes Depth Injection Module (MLP + Concat) for Stage 1/2.
    - 'student': Removes Depth Injection, adds Auxiliary Depth Regression Head for Stage 3.
    """

    def __init__(self, mode="teacher"):
        super().__init__()
        self.mode = mode

        # ---------------------------------------------------------------------
        # Encoder: ResNet34
        # ---------------------------------------------------------------------
        # Load pretrained weights
        weights = ResNet34_Weights.IMAGENET1K_V1
        backbone = resnet34(weights=weights)

        # Modify first convolution to accept 1-channel input (Grayscale)
        # We sum the weights across the channel dimension to preserve intensity info
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # Out: 64ch, H/4
        self.layer2 = backbone.layer2  # Out: 128ch, H/8
        self.layer3 = backbone.layer3  # Out: 256ch, H/16
        self.layer4 = backbone.layer4  # Out: 512ch, H/32

        # ---------------------------------------------------------------------
        # Mode-Specific Modules
        # ---------------------------------------------------------------------
        if self.mode == "teacher":
            # Depth Injection Module
            # Projects scalar depth to feature vector and concatenates to bottleneck
            self.depth_embed_dim = 64
            self.depth_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.depth_embed_dim),
                nn.ReLU(inplace=True),
            )
            bottleneck_in = 512 + self.depth_embed_dim
        else:
            # Student Mode
            bottleneck_in = 512

            # Auxiliary Depth Head (Regression)
            # Forces encoder to learn depth-correlated features
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

        # ---------------------------------------------------------------------
        # Decoder: Wide-LinkNet
        # ---------------------------------------------------------------------
        # Dec4: H/32 -> H/16. Skip: Layer3 (256)
        self.dec4 = DecoderBlock(bottleneck_in, 256)

        # Dec3: H/16 -> H/8. Skip: Layer2 (128)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: H/8 -> H/4. Skip: Layer1 (64)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: H/4 -> H/2. Skip: Stem (64)
        # Note: Stem is the output of relu before maxpool
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsample: H/2 -> H.
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, depth=None):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            depth: Depth tensor (B, 1) - Required for Teacher, Optional for Student (unused)
        """
        # 1. Encoder
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # Stem: 64ch, H/2

        x1 = self.maxpool(x0)  # H/4
        x1 = self.layer1(x1)  # 64ch
        x2 = self.layer2(x1)  # 128ch
        x3 = self.layer3(x2)  # 256ch
        x4 = self.layer4(x3)  # 512ch

        # 2. Bottleneck & Mode Specifics
        if self.mode == "teacher":
            if depth is None:
                raise ValueError("Depth input is required for Teacher model")

            # Depth Injection
            d_feat = self.depth_mlp(depth)  # (B, 64)
            # Expand spatially to match bottleneck dimensions
            d_feat = d_feat.unsqueeze(2).unsqueeze(3)
            d_feat = d_feat.expand(-1, -1, x4.size(2), x4.size(3))

            x_bottleneck = torch.cat([x4, d_feat], dim=1)  # (B, 576, H/32, W/32)
            aux_out = None

        else:  # Student
            x_bottleneck = x4
            aux_out = self.aux_head(x4)  # (B, 1)

        # 3. Decoder
        d4 = self.dec4(x_bottleneck, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        logits = self.final_conv(d1)

        if self.mode == "student":
            return {"mask": logits, "depth": aux_out}
        else:
            return logits


def build_model(stage="teacher"):
    """
    Factory function to build the model based on the training stage.

    Args:
        stage (str): 'teacher' for Stage 1/2 (Specialist), 'student' for Stage 3 (Generalist).

    Returns:
        SaltModel: The configured model instance.
    """
    if stage == "teacher":
        return SaltModel(mode="teacher")
    elif stage == "student":
        return SaltModel(mode="student")
    else:
        raise ValueError(f"Unknown stage: {stage}")
