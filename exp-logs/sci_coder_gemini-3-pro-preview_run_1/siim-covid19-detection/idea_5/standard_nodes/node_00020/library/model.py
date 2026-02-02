import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import NUM_CLASSES


class ConvBlock(nn.Module):
    """
    Standard Double Convolution Block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """
    Standard Decoder Block.
    Flow: Upsample(Input) -> Concat(Skip) -> ConvBlock
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        # x: input from previous decoder layer (coarser)
        # skip: skip connection from encoder (finer)

        # 1. Upsample input
        x_up = self.up(x)

        # 2. Handle potential padding issues (if odd dimensions)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        # 3. Concatenate
        x_cat = torch.cat([x_up, skip], dim=1)

        # 4. Convolve
        return self.conv(x_cat)


class ResNet18UNet(nn.Module):
    """
    Multi-Task U-Net with ResNet18 Encoder (Standard U-Net, no Attention).
    Cite solution_lesson_node_00015: Attention Gates degrade performance on diffuse targets.
    Outputs:
        1. Class Logits (Study Level)
        2. Segmentation Logits (Image Level)
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # --- Encoder (ResNet18) ---
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        self.initial = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1  # 64 channels
        self.layer2 = base_model.layer2  # 128 channels
        self.layer3 = base_model.layer3  # 256 channels
        self.layer4 = base_model.layer4  # 512 channels (Bottleneck)

        # --- Classification Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_class = nn.Linear(512, num_classes)

        # --- Decoder (Standard U-Net) ---
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)

        # --- Segmentation Head ---
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # Skip for Up2
        x2 = self.layer2(x1)  # Skip for Up3
        x3 = self.layer3(x2)  # Skip for Up4
        x4 = self.layer4(x3)  # Bottleneck

        # --- Classification Branch ---
        pool = self.avgpool(x4)
        flat = torch.flatten(pool, 1)
        class_logits = self.fc_class(flat)

        # --- Decoder Path ---
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        # --- Segmentation Branch ---
        seg_logits = self.final_conv(d1)
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return class_logits, seg_logits
