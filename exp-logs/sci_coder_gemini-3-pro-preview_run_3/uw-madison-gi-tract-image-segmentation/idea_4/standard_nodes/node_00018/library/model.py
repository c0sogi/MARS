import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from library.config import (
    BACKBONE,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    DEEP_SUPERVISION,
)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    (Conv3x3 -> BN -> ReLU) x 2
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SegmentationHead(nn.Module):
    """
    1x1 Convolution to map features to class logits.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=0
        )

    def forward(self, x):
        return self.conv(x)


class UnetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with EfficientNet Encoder and Deep Supervision.

    Nodes are indexed as X^{i,j} where i is the downsampling level and j is the dense block index.
    i = 0..4 (corresponding to strides 2, 4, 8, 16, 32)
    j = 0 (Encoder), 1..4 (Decoder)
    """

    def __init__(
        self,
        backbone=BACKBONE,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        deep_supervision=DEEP_SUPERVISION,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder
        # Using timm to create the backbone
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            backbone,
            pretrained=(encoder_weights is not None),
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),  # Strides: 2, 4, 8, 16, 32
        )

        # Get channel counts from the encoder
        # Dummy forward pass to get shapes or use feature_info
        feature_info = self.encoder.feature_info
        enc_channels = (
            feature_info.channels()
        )  # e.g. [24, 32, 56, 160, 448] for effnet-b4

        # Define decoder channel widths
        # We can keep them same as encoder or use a fixed list.
        # Here we define them somewhat arbitrarily but consistent with U-Net style reduction
        # For U-Net++, standard practice is often to keep channels consistent with the skip level
        self.filters = enc_channels

        # 2. Decoder Nodes
        # Naming convention: conv{i}_{j} implements node X^{i,j}

        # --- Column j=1 ---
        # Input: [X^{i,0}, Up(X^{i+1, 0})]
        self.conv0_1 = ConvBlock(self.filters[0] + self.filters[1], self.filters[0])
        self.conv1_1 = ConvBlock(self.filters[1] + self.filters[2], self.filters[1])
        self.conv2_1 = ConvBlock(self.filters[2] + self.filters[3], self.filters[2])
        self.conv3_1 = ConvBlock(self.filters[3] + self.filters[4], self.filters[3])

        # --- Column j=2 ---
        # Input: [X^{i,0}, X^{i,1}, Up(X^{i+1, 1})]
        self.conv0_2 = ConvBlock(self.filters[0] * 2 + self.filters[1], self.filters[0])
        self.conv1_2 = ConvBlock(self.filters[1] * 2 + self.filters[2], self.filters[1])
        self.conv2_2 = ConvBlock(self.filters[2] * 2 + self.filters[3], self.filters[2])

        # --- Column j=3 ---
        # Input: [X^{i,0}, X^{i,1}, X^{i,2}, Up(X^{i+1, 2})]
        self.conv0_3 = ConvBlock(self.filters[0] * 3 + self.filters[1], self.filters[0])
        self.conv1_3 = ConvBlock(self.filters[1] * 3 + self.filters[2], self.filters[1])

        # --- Column j=4 ---
        # Input: [X^{i,0}, X^{i,1}, X^{i,2}, X^{i,3}, Up(X^{i+1, 3})]
        self.conv0_4 = ConvBlock(self.filters[0] * 4 + self.filters[1], self.filters[0])

        # 3. Segmentation Heads
        # We attach heads to X^{0,j} for j=1..4
        self.final_head = SegmentationHead(self.filters[0], classes)

        if self.deep_supervision:
            self.head1 = SegmentationHead(self.filters[0], classes)
            self.head2 = SegmentationHead(self.filters[0], classes)
            self.head3 = SegmentationHead(self.filters[0], classes)

    def _upsample_add(self, x, y):
        """
        Upsample x to match y's shape and concatenate.
        x: tensor to be upsampled (from lower level)
        y: list of tensors to concatenate (from same level)
        """
        _, _, h, w = y[0].shape

        # Bilinear upsampling
        x_up = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        # Concatenate all
        return torch.cat([*y, x_up], dim=1)

    def forward(self, x):
        # 1. Encoder Pass
        features = self.encoder(x)

        x0_0 = features[0]  # Stride 2
        x1_0 = features[1]  # Stride 4
        x2_0 = features[2]  # Stride 8
        x3_0 = features[3]  # Stride 16
        x4_0 = features[4]  # Stride 32

        # 2. Decoder Pass

        # --- Column 1 ---
        x0_1 = self.conv0_1(self._upsample_add(x1_0, [x0_0]))
        x1_1 = self.conv1_1(self._upsample_add(x2_0, [x1_0]))
        x2_1 = self.conv2_1(self._upsample_add(x3_0, [x2_0]))
        x3_1 = self.conv3_1(self._upsample_add(x4_0, [x3_0]))

        # --- Column 2 ---
        x0_2 = self.conv0_2(self._upsample_add(x1_1, [x0_0, x0_1]))
        x1_2 = self.conv1_2(self._upsample_add(x2_1, [x1_0, x1_1]))
        x2_2 = self.conv2_2(self._upsample_add(x3_1, [x2_0, x2_1]))

        # --- Column 3 ---
        x0_3 = self.conv0_3(self._upsample_add(x1_2, [x0_0, x0_1, x0_2]))
        x1_3 = self.conv1_3(self._upsample_add(x2_2, [x1_0, x1_1, x1_2]))

        # --- Column 4 ---
        x0_4 = self.conv0_4(self._upsample_add(x1_3, [x0_0, x0_1, x0_2, x0_3]))

        # 3. Output Heads
        # The output features are at stride 2. We need to upsample to original input size (stride 1).
        # However, standard U-Net usually outputs at the resolution of the first encoder block (stride 2)
        # and relies on final interpolation or assumes input was padded.
        # Given the task, we usually interpolate the logits to the original image size.

        logits_final = self.final_head(x0_4)
        logits_final = F.interpolate(
            logits_final, size=x.shape[2:], mode="bilinear", align_corners=True
        )

        if self.training and self.deep_supervision:
            logits1 = self.head1(x0_1)
            logits1 = F.interpolate(
                logits1, size=x.shape[2:], mode="bilinear", align_corners=True
            )

            logits2 = self.head2(x0_2)
            logits2 = F.interpolate(
                logits2, size=x.shape[2:], mode="bilinear", align_corners=True
            )

            logits3 = self.head3(x0_3)
            logits3 = F.interpolate(
                logits3, size=x.shape[2:], mode="bilinear", align_corners=True
            )

            return [logits_final, logits1, logits2, logits3]

        return logits_final
