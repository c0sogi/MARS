import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard building block for U-Net++ decoder nodes.
    Consists of two 3x3 convolutions, each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UnetPlusPlus(nn.Module):
    """
    2D U-Net++ with ResNet-34 Backbone and Weighted Deep Supervision.
    Modified to accept 4-channel input (RGB + Depth Map).
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        # Load pretrained weights
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.backbone = resnet34(weights=weights)

        # Modify first layer for 4 channels
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 4
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize 4th channel with mean of RGB weights
            new_conv.weight[:, 3:4, :, :] = torch.mean(
                old_conv.weight, dim=1, keepdim=True
            )

        self.backbone.conv1 = new_conv

        # Channel counts for ResNet34 stages
        # x0: conv1 (Stride 2) -> 64
        # x1: layer1 (Stride 4) -> 64
        # x2: layer2 (Stride 8) -> 128
        # x3: layer3 (Stride 16) -> 256
        # x4: layer4 (Stride 32) -> 512
        filters = [64, 64, 128, 256, 512]

        # 2. Decoder Blocks (Nested U-Net Structure)
        # Naming convention: conv{level}_{depth}
        # Level 0 (Stride 2), Level 1 (Stride 4), etc.

        # Column 1 (Depth 1)
        # Inputs: Up(X_{i+1, 0}) + X_{i, 0}
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        # Column 2 (Depth 2)
        # Inputs: Up(X_{i+1, 1}) + X_{i, 0} + X_{i, 1}
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        # Column 3 (Depth 3)
        # Inputs: Up(X_{i+1, 2}) + X_{i, 0} + X_{i, 1} + X_{i, 2}
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        # Column 4 (Depth 4) - Final
        # Inputs: Up(X_{i+1, 3}) + X_{i, 0} + X_{i, 1} + X_{i, 2} + X_{i, 3}
        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        # 3. Segmentation Heads (Deep Supervision)
        # We attach heads to X_{0,1}, X_{0,2}, X_{0,3}, X_{0,4}
        self.final_head = nn.Conv2d(filters[0], Config.NUM_CLASSES, kernel_size=1)
        self.aux_head1 = nn.Conv2d(
            filters[0], Config.NUM_CLASSES, kernel_size=1
        )  # From X_{0,3}
        self.aux_head2 = nn.Conv2d(
            filters[0], Config.NUM_CLASSES, kernel_size=1
        )  # From X_{0,2}
        self.aux_head3 = nn.Conv2d(
            filters[0], Config.NUM_CLASSES, kernel_size=1
        )  # From X_{0,1}

    def forward(self, x):
        input_size = x.shape[-2:]

        # --- Encoder ---
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x0 = self.backbone.relu(x)  # Stride 2, 64

        x = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x)  # Stride 4, 64
        x2 = self.backbone.layer2(x1)  # Stride 8, 128
        x3 = self.backbone.layer3(x2)  # Stride 16, 256
        x4 = self.backbone.layer4(x3)  # Stride 32, 512

        # --- Decoder ---

        # Column 1
        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0,
                    F.interpolate(
                        x1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1,
                    F.interpolate(
                        x2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2,
                    F.interpolate(
                        x3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3,
                    F.interpolate(
                        x4, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 2
        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0,
                    x0_1,
                    F.interpolate(
                        x1_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1,
                    x1_1,
                    F.interpolate(
                        x2_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2,
                    x2_1,
                    F.interpolate(
                        x3_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 3
        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0,
                    x0_1,
                    x0_2,
                    F.interpolate(
                        x1_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1,
                    x1_1,
                    x1_2,
                    F.interpolate(
                        x2_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 4 (Final)
        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0,
                    x0_1,
                    x0_2,
                    x0_3,
                    F.interpolate(
                        x1_3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Heads ---
        # All outputs are currently at Stride 2 (x0 resolution).
        # We need to upsample to input resolution (Stride 1).

        logits_final = self.final_head(x0_4)
        logits_aux1 = self.aux_head1(x0_3)
        logits_aux2 = self.aux_head2(x0_2)
        logits_aux3 = self.aux_head3(x0_1)

        # Upsample to original image size
        logits_final = F.interpolate(
            logits_final, size=input_size, mode="bilinear", align_corners=True
        )

        if self.training and Config.DEEP_SUPERVISION:
            logits_aux1 = F.interpolate(
                logits_aux1, size=input_size, mode="bilinear", align_corners=True
            )
            logits_aux2 = F.interpolate(
                logits_aux2, size=input_size, mode="bilinear", align_corners=True
            )
            logits_aux3 = F.interpolate(
                logits_aux3, size=input_size, mode="bilinear", align_corners=True
            )

            # Return list matching DS_WEIGHTS: [Final, Aux1, Aux2, Aux3]
            return [logits_final, logits_aux1, logits_aux2, logits_aux3]

        return logits_final
