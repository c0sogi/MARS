import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Enhances important features by suppressing background noise.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block with Batch Normalization, ReLU, and scSE Attention.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.scse(x)
        return x


class UpBlock(nn.Module):
    """
    U-Net++ Nested Decoder Node.
    Receives an upsampled feature map from the lower level and a list of
    skip connection feature maps from the same level (dense connections).
    """

    def __init__(self, in_channels_up, in_channels_skip_list, out_channels):
        super().__init__()
        # Total input channels = upsampled_channels + sum(skip_channels)
        total_in_channels = in_channels_up + sum(in_channels_skip_list)
        self.conv = ConvBlock(total_in_channels, out_channels)

    def forward(self, x_up, x_skips):
        """
        Args:
            x_up: Tensor from lower level (i+1, j-1)
            x_skips: List of Tensors from same level (i, 0..j-1)
        """
        # Upsample x_up to match the spatial resolution of x_skips
        target_size = x_skips[0].shape[2:]
        x_up = F.interpolate(
            x_up, size=target_size, mode="bilinear", align_corners=True
        )

        # Concatenate along channel dimension
        x = torch.cat([x_up] + x_skips, dim=1)
        return self.conv(x)


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 (32x4d) Encoder and scSE Attention.
    Implements Deep Supervision and Stride 2 Output termination.
    """

    def __init__(
        self,
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.CHANNELS,
        classes=1,
    ):
        super().__init__()

        # 1. Encoder Initialization (timm)
        # ResNeXt-50 features at indices:
        # 0: Stride 2 (64 ch)
        # 1: Stride 4 (256 ch)
        # 2: Stride 8 (512 ch)
        # 3: Stride 16 (1024 ch)
        # 4: Stride 32 (2048 ch)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder output channels
        enc_dims = self.encoder.feature_info.channels()

        # Decoder channels (reversed Config to match levels 0->4)
        # Level 0 (Stride 2): 16
        # Level 1 (Stride 4): 32
        # ...
        # Level 4 (Stride 32): 256
        dec_dims = list(Config.DECODER_CHANNELS)[::-1]  # [16, 32, 64, 128, 256]

        # 2. Projection Layers (Adapting Encoder to Decoder Nodes X^i,0)
        self.conv0_0 = ConvBlock(enc_dims[0], dec_dims[0])
        self.conv1_0 = ConvBlock(enc_dims[1], dec_dims[1])
        self.conv2_0 = ConvBlock(enc_dims[2], dec_dims[2])
        self.conv3_0 = ConvBlock(enc_dims[3], dec_dims[3])
        self.conv4_0 = ConvBlock(enc_dims[4], dec_dims[4])

        # 3. Decoder Nodes (Nested Skip Connections)

        # Column 1 (j=1)
        self.conv3_1 = UpBlock(dec_dims[4], [dec_dims[3]], dec_dims[3])
        self.conv2_1 = UpBlock(dec_dims[3], [dec_dims[2]], dec_dims[2])
        self.conv1_1 = UpBlock(dec_dims[2], [dec_dims[1]], dec_dims[1])
        self.conv0_1 = UpBlock(dec_dims[1], [dec_dims[0]], dec_dims[0])

        # Column 2 (j=2)
        self.conv2_2 = UpBlock(dec_dims[3], [dec_dims[2], dec_dims[2]], dec_dims[2])
        self.conv1_2 = UpBlock(dec_dims[2], [dec_dims[1], dec_dims[1]], dec_dims[1])
        self.conv0_2 = UpBlock(dec_dims[1], [dec_dims[0], dec_dims[0]], dec_dims[0])

        # Column 3 (j=3)
        self.conv1_3 = UpBlock(
            dec_dims[2], [dec_dims[1], dec_dims[1], dec_dims[1]], dec_dims[1]
        )
        self.conv0_3 = UpBlock(
            dec_dims[1], [dec_dims[0], dec_dims[0], dec_dims[0]], dec_dims[0]
        )

        # Column 4 (j=4) - Final Node
        self.conv0_4 = UpBlock(
            dec_dims[1],
            [dec_dims[0], dec_dims[0], dec_dims[0], dec_dims[0]],
            dec_dims[0],
        )

        # 4. Deep Supervision Heads
        # Attached to all Level 0 nodes (Stride 2)
        self.seg1 = nn.Conv2d(dec_dims[0], classes, 1)
        self.seg2 = nn.Conv2d(dec_dims[0], classes, 1)
        self.seg3 = nn.Conv2d(dec_dims[0], classes, 1)
        self.seg4 = nn.Conv2d(dec_dims[0], classes, 1)

    def forward(self, x):
        # 1. Encoder Pass
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features

        # 2. Decoder Pass

        # Column 0 (Projections)
        x0_0 = self.conv0_0(x0)
        x1_0 = self.conv1_0(x1)
        x2_0 = self.conv2_0(x2)
        x3_0 = self.conv3_0(x3)
        x4_0 = self.conv4_0(x4)

        # Column 1
        x3_1 = self.conv3_1(x4_0, [x3_0])
        x2_1 = self.conv2_1(x3_0, [x2_0])
        x1_1 = self.conv1_1(x2_0, [x1_0])
        x0_1 = self.conv0_1(x1_0, [x0_0])

        # Column 2
        x2_2 = self.conv2_2(x3_1, [x2_0, x2_1])
        x1_2 = self.conv1_2(x2_1, [x1_0, x1_1])
        x0_2 = self.conv0_2(x1_1, [x0_0, x0_1])

        # Column 3
        x1_3 = self.conv1_3(x2_2, [x1_0, x1_1, x1_2])
        x0_3 = self.conv0_3(x1_2, [x0_0, x0_1, x0_2])

        # Column 4
        x0_4 = self.conv0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])

        # 3. Output Generation
        # Helper to upsample logits from Stride 2 to Input Resolution (Stride 1)
        def final_up(t):
            return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=True)

        out1 = final_up(self.seg1(x0_1))
        out2 = final_up(self.seg2(x0_2))
        out3 = final_up(self.seg3(x0_3))
        out4 = final_up(self.seg4(x0_4))

        if self.training:
            # Return list for Deep Supervision loss calculation
            return [out1, out2, out3, out4]
        else:
            # Return final output for inference
            return out4
