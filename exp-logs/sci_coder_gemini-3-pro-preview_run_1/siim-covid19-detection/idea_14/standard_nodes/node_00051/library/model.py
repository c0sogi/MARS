import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import cfg


class ResNet18D_UNet(nn.Module):
    """
    ResNet18-D Multi-Task U-Net.

    This architecture combines a ResNet18-D (Deep Stem) backbone for feature extraction
    with a U-Net style decoder for dense prediction. It simultaneously predicts:
    1. Study-level labels (Negative, Typical, Indeterminate, Atypical)
    2. Image-level segmentation masks (Opacity)

    Attributes:
        encoder (nn.Module): ResNet18-D backbone from timm.
        cls_head (nn.Module): Classification head (GAP + Linear).
        decoder (nn.Module): Series of upsampling and convolution blocks.
        seg_head (nn.Module): Final 1x1 convolution for segmentation mask.
    """

    def __init__(self):
        super(ResNet18D_UNet, self).__init__()

        # =================================================================
        # 1. Encoder: ResNet18-D
        # =================================================================
        # We use 'features_only=True' to extract intermediate feature maps for the U-Net skip connections.
        # out_indices=(0, 1, 2, 3, 4) corresponds to features at strides 2, 4, 8, 16, 32.
        # ResNet18-D replaces the standard 7x7 stem with three 3x3 convolutions, preserving spatial info.
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Retrieve channel counts for the specific backbone (usually [64, 64, 128, 256, 512] for ResNet18)
        enc_channels = self.encoder.feature_info.channels()

        # =================================================================
        # 2. Classification Head
        # =================================================================
        # Strategy: Global Average Pooling (GAP) -> Flatten -> Linear Layer
        # We use the deepest feature map (stride 32) which contains the most semantic information.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(enc_channels[-1], cfg.num_study_classes)

        # =================================================================
        # 3. Decoder: U-Net Style
        # =================================================================
        # We construct the decoder by upsampling and concatenating with encoder skip connections.

        # Block 1: Stride 32 -> 16
        # Input: Layer 4 (512 ch) -> Up -> Concat Layer 3 (256 ch) -> Conv
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = self._make_decoder_block(enc_channels[4] + enc_channels[3], 256)

        # Block 2: Stride 16 -> 8
        # Input: Prev (256 ch) -> Up -> Concat Layer 2 (128 ch) -> Conv
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = self._make_decoder_block(256 + enc_channels[2], 128)

        # Block 3: Stride 8 -> 4
        # Input: Prev (128 ch) -> Up -> Concat Layer 1 (64 ch) -> Conv
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = self._make_decoder_block(128 + enc_channels[1], 64)

        # Block 4: Stride 4 -> 2
        # Input: Prev (64 ch) -> Up -> Concat Layer 0 (64 ch) -> Conv
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = self._make_decoder_block(64 + enc_channels[0], 32)

        # Block 5: Stride 2 -> 1 (Original Resolution)
        # Input: Prev (32 ch) -> Up -> Conv (No skip connection available at stride 1)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5 = self._make_decoder_block(32, 16)

        # =================================================================
        # 4. Segmentation Head
        # =================================================================
        # Projects final features to the number of segmentation classes (1 for Opacity)
        self.seg_head = nn.Conv2d(16, cfg.num_seg_classes, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        """
        Creates a standard U-Net decoder block:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            cls_logits (torch.Tensor): Study-level logits (B, num_study_classes).
            seg_logits (torch.Tensor): Segmentation logits (B, num_seg_classes, H, W).
        """
        # --- Encoder ---
        # features is a list of tensors [f0, f1, f2, f3, f4]
        # f0: stride 2, f1: stride 4, f2: stride 8, f3: stride 16, f4: stride 32
        features = self.encoder(x)

        # --- Classification Branch ---
        # Apply GAP to the deepest feature map (f4)
        global_feat = self.global_pool(features[-1])  # (B, C, 1, 1)
        global_feat = torch.flatten(global_feat, 1)  # (B, C)
        cls_logits = self.cls_head(global_feat)  # (B, num_classes)

        # --- Segmentation Branch (Decoder) ---

        # 1. Upsample f4 and concatenate with f3
        x_dec = self.up1(features[4])
        # Robustness check: Ensure spatial dimensions match before concatenation
        if x_dec.size()[2:] != features[3].size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=features[3].shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, features[3]], dim=1)
        x_dec = self.conv1(x_dec)

        # 2. Upsample and concatenate with f2
        x_dec = self.up2(x_dec)
        if x_dec.size()[2:] != features[2].size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=features[2].shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, features[2]], dim=1)
        x_dec = self.conv2(x_dec)

        # 3. Upsample and concatenate with f1
        x_dec = self.up3(x_dec)
        if x_dec.size()[2:] != features[1].size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=features[1].shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, features[1]], dim=1)
        x_dec = self.conv3(x_dec)

        # 4. Upsample and concatenate with f0
        x_dec = self.up4(x_dec)
        if x_dec.size()[2:] != features[0].size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=features[0].shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, features[0]], dim=1)
        x_dec = self.conv4(x_dec)

        # 5. Final Upsample to original resolution
        x_dec = self.up5(x_dec)
        # Ensure output matches input resolution exactly
        if x_dec.size()[2:] != x.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=x.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = self.conv5(x_dec)

        # Generate segmentation logits
        seg_logits = self.seg_head(x_dec)

        return cls_logits, seg_logits
