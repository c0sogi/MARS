import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.utils import CFG


class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    Used for the decoder nodes in U-Net++.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetPlusPlus(nn.Module):
    """
    Nested U-Net (U-Net++) with EfficientNet Backbone and Deep Supervision.
    """

    def __init__(
        self,
        backbone_name=CFG.backbone,
        in_channels=3,
        classes=CFG.num_classes,
        pretrained=True,
    ):
        super().__init__()

        # --- Encoder (Backbone) ---
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )

        # Determine encoder channel counts dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            features = self.encoder(dummy)
            enc_channels = [f.shape[1] for f in features]
            # EfficientNet-B4 typically returns 5 feature maps with strides [2, 4, 8, 16, 32]

        # --- Decoder Configuration ---
        # Define channel counts for decoder nodes at levels 0, 1, 2, 3, 4
        # Level 0 corresponds to stride 2 (matching enc_channels[0])
        self.dec_channels = [32, 64, 128, 256, 320]

        # --- Decoder Blocks (Nested Skip Connections) ---
        # Notation: conv_i_j where i is the scale level (row) and j is the dense block index (column)

        # Column 1 (j=1): Receives Encoder[i] and Upsampled Encoder[i+1]
        self.conv_0_1 = ConvBlock(
            enc_channels[0] + enc_channels[1], self.dec_channels[0]
        )
        self.conv_1_1 = ConvBlock(
            enc_channels[1] + enc_channels[2], self.dec_channels[1]
        )
        self.conv_2_1 = ConvBlock(
            enc_channels[2] + enc_channels[3], self.dec_channels[2]
        )
        self.conv_3_1 = ConvBlock(
            enc_channels[3] + enc_channels[4], self.dec_channels[3]
        )

        # Column 2 (j=2): Receives Encoder[i], Node[i,1], and Upsampled Node[i+1,1]
        self.conv_0_2 = ConvBlock(
            enc_channels[0] + self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )
        self.conv_1_2 = ConvBlock(
            enc_channels[1] + self.dec_channels[1] + self.dec_channels[2],
            self.dec_channels[1],
        )
        self.conv_2_2 = ConvBlock(
            enc_channels[2] + self.dec_channels[2] + self.dec_channels[3],
            self.dec_channels[2],
        )

        # Column 3 (j=3): Receives Encoder[i], Node[i,1], Node[i,2], and Upsampled Node[i+1,2]
        self.conv_0_3 = ConvBlock(
            enc_channels[0] + 2 * self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )
        self.conv_1_3 = ConvBlock(
            enc_channels[1] + 2 * self.dec_channels[1] + self.dec_channels[2],
            self.dec_channels[1],
        )

        # Column 4 (j=4): Receives Encoder[i], Node[i,1], Node[i,2], Node[i,3], and Upsampled Node[i+1,3]
        self.conv_0_4 = ConvBlock(
            enc_channels[0] + 3 * self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )

        # --- Deep Supervision Heads ---
        # Attached to x0_1, x0_2, x0_3, x0_4
        self.head1 = nn.Conv2d(self.dec_channels[0], classes, kernel_size=1)
        self.head2 = nn.Conv2d(self.dec_channels[0], classes, kernel_size=1)
        self.head3 = nn.Conv2d(self.dec_channels[0], classes, kernel_size=1)
        self.head4 = nn.Conv2d(self.dec_channels[0], classes, kernel_size=1)

    def _up(self, x, target):
        """Upsamples tensor x to match the spatial dimensions of target."""
        if x.shape[-2:] != target.shape[-2:]:
            return F.interpolate(
                x, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def forward(self, x):
        input_size = x.shape[-2:]

        # --- Encoder ---
        features = self.encoder(x)
        x0_0 = features[0]  # stride 2
        x1_0 = features[1]  # stride 4
        x2_0 = features[2]  # stride 8
        x3_0 = features[3]  # stride 16
        x4_0 = features[4]  # stride 32

        # --- Decoder Column 1 ---
        x0_1 = self.conv_0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv_1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv_2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv_3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        # --- Decoder Column 2 ---
        x0_2 = self.conv_0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv_1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv_2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        # --- Decoder Column 3 ---
        x0_3 = self.conv_0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv_1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        # --- Decoder Column 4 ---
        x0_4 = self.conv_0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1)
        )

        # --- Heads & Upsampling ---
        # All heads are at stride 2 (resolution of x0_0). Upsample to input resolution (stride 1).
        out1 = self.head1(x0_1)
        out2 = self.head2(x0_2)
        out3 = self.head3(x0_3)
        out4 = self.head4(x0_4)

        out1 = F.interpolate(
            out1, size=input_size, mode="bilinear", align_corners=False
        )
        out2 = F.interpolate(
            out2, size=input_size, mode="bilinear", align_corners=False
        )
        out3 = F.interpolate(
            out3, size=input_size, mode="bilinear", align_corners=False
        )
        out4 = F.interpolate(
            out4, size=input_size, mode="bilinear", align_corners=False
        )

        if self.training:
            # Return all heads for Deep Supervision loss
            return [out4, out3, out2, out1]
        else:
            # Return only the final refined output for inference
            return out4
