import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat with Skip -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # The input to conv1 is the concatenation of upsampled input and skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # 1. Upsample the input from the lower level
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # 2. Concatenate with skip connection
        if skip is not None:
            # Handle potential slight shape mismatches (e.g. due to odd padding in encoder)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # 3. Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MultiTaskUNet(nn.Module):
    """
    Unified Encoder-Decoder architecture for simultaneous Study Classification
    and Opacity Segmentation.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=True,
        num_study_classes=Config.NUM_STUDY_CLASSES,
        num_image_classes=Config.NUM_IMAGE_CLASSES,
    ):
        super().__init__()

        # --- Encoder ---
        # features_only=True returns a list of feature maps at different scales
        # out_indices=(0, 1, 2, 3, 4) corresponds to strides 2, 4, 8, 16, 32 for ResNet
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get the number of channels for each feature map
        # Example for ResNet18: [64, 64, 128, 256, 512]
        feature_channels = self.encoder.feature_info.channels()

        # --- Classification Head (Study Level) ---
        # Attached to the bottleneck (deepest feature map, stride 32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(feature_channels[-1], num_study_classes)

        # --- Decoder (Segmentation Level) ---
        # We define output channels for each decoder step
        dec_channels = [256, 128, 64, 64]

        # Block 4: Input from Bottleneck (f4), Skip from f3
        self.dec4 = DecoderBlock(
            feature_channels[4], feature_channels[3], dec_channels[0]
        )

        # Block 3: Input from dec4, Skip from f2
        self.dec3 = DecoderBlock(dec_channels[0], feature_channels[2], dec_channels[1])

        # Block 2: Input from dec3, Skip from f1
        self.dec2 = DecoderBlock(dec_channels[1], feature_channels[1], dec_channels[2])

        # Block 1: Input from dec2, Skip from f0
        self.dec1 = DecoderBlock(dec_channels[2], feature_channels[0], dec_channels[3])

        # --- Segmentation Head ---
        self.seg_head = nn.Conv2d(dec_channels[3], num_image_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (N, 3, H, W)

        Returns:
            cls_logits (torch.Tensor): Study-level logits (N, num_study_classes)
            mask_logits (torch.Tensor): Segmentation logits (N, num_image_classes, H, W)
        """
        input_size = x.shape[-2:]

        # 1. Encoder Pass
        features = self.encoder(x)
        # features list: [f0 (s2), f1 (s4), f2 (s8), f3 (s16), f4 (s32)]

        # 2. Classification Branch
        # Use the deepest feature map (f4)
        x_cls = self.global_pool(features[-1])
        x_cls = torch.flatten(x_cls, 1)
        cls_logits = self.cls_head(x_cls)

        # 3. Segmentation Branch (Decoder)
        # Upsample from bottleneck back to higher resolutions
        x_dec = self.dec4(features[4], features[3])
        x_dec = self.dec3(x_dec, features[2])
        x_dec = self.dec2(x_dec, features[1])
        x_dec = self.dec1(x_dec, features[0])

        # 4. Final Segmentation Output
        mask_logits = self.seg_head(x_dec)

        # Upsample to original input resolution (dec1 output is stride 2)
        mask_logits = F.interpolate(
            mask_logits, size=input_size, mode="bilinear", align_corners=True
        )

        return cls_logits, mask_logits
