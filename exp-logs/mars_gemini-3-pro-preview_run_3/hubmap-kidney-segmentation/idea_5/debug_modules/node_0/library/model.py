import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvNeXtUNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with ConvNeXt-Tiny backbone and Deep Supervision.

    Returns a list of outputs from 4 semantic levels (scales) to support
    weighted loss calculation: [Fine, Medium-Fine, Medium-Coarse, Coarse].
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # 1. Encoder: ConvNeXt Tiny
        # features_only=True returns a list of feature maps at strides [4, 8, 16, 32]
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=pretrained, features_only=True
        )

        # Get channel counts from the backbone
        # Typically [96, 192, 384, 768] for convnext_tiny
        dims = self.encoder.feature_info.channels()
        c0, c1, c2, c3 = dims

        # 2. Decoder: U-Net++ Nested Skip Pathways
        # We maintain the same channel width as the encoder at each level for simplicity.

        # --- Level 1 (Stride 16) Nodes ---
        # x_2_1: Inputs from x_2_0 (c2) and upsampled x_3_0 (c3)
        self.conv_2_1 = ConvBlock(c2 + c3, c2)

        # --- Level 2 (Stride 8) Nodes ---
        # x_1_1: Inputs from x_1_0 (c1) and upsampled x_2_0 (c2)
        self.conv_1_1 = ConvBlock(c1 + c2, c1)
        # x_1_2: Inputs from x_1_0 (c1), x_1_1 (c1), and upsampled x_2_1 (c2)
        self.conv_1_2 = ConvBlock(c1 * 2 + c2, c1)

        # --- Level 3 (Stride 4) Nodes ---
        # x_0_1: Inputs from x_0_0 (c0) and upsampled x_1_0 (c1)
        self.conv_0_1 = ConvBlock(c0 + c1, c0)
        # x_0_2: Inputs from x_0_0 (c0), x_0_1 (c0), and upsampled x_1_1 (c1)
        self.conv_0_2 = ConvBlock(c0 * 2 + c1, c0)
        # x_0_3: Inputs from x_0_0 (c0), x_0_1 (c0), x_0_2 (c0), and upsampled x_1_2 (c1)
        self.conv_0_3 = ConvBlock(c0 * 3 + c1, c0)

        # 3. Deep Supervision Heads
        # We attach heads to the output of each skip pathway level to enable multi-scale supervision.
        # Head 0: Finest resolution (Stride 4) -> x_0_3
        self.head0 = nn.Conv2d(c0, num_classes, kernel_size=1)
        # Head 1: Medium resolution (Stride 8) -> x_1_2
        self.head1 = nn.Conv2d(c1, num_classes, kernel_size=1)
        # Head 2: Coarse resolution (Stride 16) -> x_2_1
        self.head2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        # Head 3: Coarsest resolution (Stride 32) -> x_3_0 (Backbone feature)
        self.head3 = nn.Conv2d(c3, num_classes, kernel_size=1)

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]

        # --- Encoder Forward ---
        features = self.encoder(x)
        x_0_0 = features[0]  # Stride 4
        x_1_0 = features[1]  # Stride 8
        x_2_0 = features[2]  # Stride 16
        x_3_0 = features[3]  # Stride 32

        # --- Decoder Forward ---

        # J=1 Column
        # x_2_1
        u_3_0 = F.interpolate(
            x_3_0, size=x_2_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, u_3_0], dim=1))

        # x_1_1
        u_2_0 = F.interpolate(
            x_2_0, size=x_1_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, u_2_0], dim=1))

        # x_0_1
        u_1_0 = F.interpolate(
            x_1_0, size=x_0_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, u_1_0], dim=1))

        # J=2 Column
        # x_1_2
        u_2_1 = F.interpolate(
            x_2_1, size=x_1_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, u_2_1], dim=1))

        # x_0_2
        u_1_1 = F.interpolate(
            x_1_1, size=x_0_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, u_1_1], dim=1))

        # J=3 Column
        # x_0_3
        u_1_2 = F.interpolate(
            x_1_2, size=x_0_0.shape[2:], mode="bilinear", align_corners=False
        )
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, u_1_2], dim=1))

        # --- Heads & Upsampling ---
        # Compute logits at each scale
        out0 = self.head0(x_0_3)
        out1 = self.head1(x_1_2)
        out2 = self.head2(x_2_1)
        out3 = self.head3(x_3_0)

        # Upsample all outputs to original image size
        out0 = F.interpolate(
            out0, size=(img_h, img_w), mode="bilinear", align_corners=False
        )
        out1 = F.interpolate(
            out1, size=(img_h, img_w), mode="bilinear", align_corners=False
        )
        out2 = F.interpolate(
            out2, size=(img_h, img_w), mode="bilinear", align_corners=False
        )
        out3 = F.interpolate(
            out3, size=(img_h, img_w), mode="bilinear", align_corners=False
        )

        # Return list for Deep Supervision loss
        return [out0, out1, out2, out3]
