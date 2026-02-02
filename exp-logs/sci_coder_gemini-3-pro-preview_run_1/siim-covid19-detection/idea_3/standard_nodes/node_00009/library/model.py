import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Used in the decoder path of the U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EfficientNetUnet(nn.Module):
    """
    Multi-Task U-Net with EfficientNet-B3 Encoder.
    Performs simultaneous Study-Level Classification and Image-Level Segmentation.
    """

    def __init__(
        self,
        encoder_name="resnet18",
        pretrained=True,
        num_study_classes=4,
        dropout_rate=0.5,
    ):
        """
        Args:
            encoder_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load ImageNet weights.
            num_study_classes (int): Number of classes for the classification head.
            dropout_rate (float): Dropout rate for the classification head.
        """
        super(EfficientNetUnet, self).__init__()

        # 1. Encoder (ResNet18 or similar)
        # features_only=True returns a list of feature maps at different strides
        # Indices (0, 1, 2, 3, 4) typically correspond to strides (2, 4, 8, 16, 32)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Determine channel counts dynamically
        # We run a dummy forward pass to get the exact shapes of the feature maps
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.encoder(dummy_input)
            self.enc_channels = [f.shape[1] for f in features]

        # 2. Classification Head
        # Attached to the deepest feature map (bottleneck, stride 32)
        # Simplified head (Cite solution_lesson_node_00008)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(self.enc_channels[-1], num_study_classes)

        # 3. Decoder
        # We upsample from stride 32 back to stride 1 (original image size)
        # Decoder channel configurations
        dec_channels = [256, 128, 64, 32, 16]

        # Block 1: Input from bottleneck (s32) -> Up to s16 -> Concat with enc_feat[3]
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = ConvBlock(
            self.enc_channels[-1] + self.enc_channels[3], dec_channels[0]
        )

        # Block 2: Input from Block 1 -> Up to s8 -> Concat with enc_feat[2]
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = ConvBlock(dec_channels[0] + self.enc_channels[2], dec_channels[1])

        # Block 3: Input from Block 2 -> Up to s4 -> Concat with enc_feat[1]
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = ConvBlock(dec_channels[1] + self.enc_channels[1], dec_channels[2])

        # Block 4: Input from Block 3 -> Up to s2 -> Concat with enc_feat[0]
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = ConvBlock(dec_channels[2] + self.enc_channels[0], dec_channels[3])

        # Block 5: Input from Block 4 -> Up to s1 (Original Size) -> No Skip (or input image skip if desired, but usually omitted)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5 = ConvBlock(dec_channels[3], dec_channels[4])

        # 4. Segmentation Head
        # Projects final decoder features to 1 channel (logits)
        self.seg_head = nn.Conv2d(dec_channels[4], 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)

        Returns:
            cls_logits (torch.Tensor): Classification logits (B, NumClasses)
            seg_logits (torch.Tensor): Segmentation logits (B, 1, H, W)
        """
        # --- Encoder ---
        # features list: [f0(s2), f1(s4), f2(s8), f3(s16), f4(s32)]
        features = self.encoder(x)
        f0, f1, f2, f3, f4 = features

        # --- Classification Branch ---
        # Global Average Pooling on the bottleneck features
        cls_feat = self.avgpool(f4)
        cls_logits = self.cls_head(cls_feat)

        # --- Segmentation Branch (Decoder) ---
        # 1. Up 32 -> 16
        x_dec = self.up1(f4)
        # Resize f3 if necessary (though usually matches if input is power of 2)
        if x_dec.size()[2:] != f3.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=f3.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, f3], dim=1)
        x_dec = self.conv1(x_dec)

        # 2. Up 16 -> 8
        x_dec = self.up2(x_dec)
        if x_dec.size()[2:] != f2.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=f2.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, f2], dim=1)
        x_dec = self.conv2(x_dec)

        # 3. Up 8 -> 4
        x_dec = self.up3(x_dec)
        if x_dec.size()[2:] != f1.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=f1.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, f1], dim=1)
        x_dec = self.conv3(x_dec)

        # 4. Up 4 -> 2
        x_dec = self.up4(x_dec)
        if x_dec.size()[2:] != f0.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=f0.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = torch.cat([x_dec, f0], dim=1)
        x_dec = self.conv4(x_dec)

        # 5. Up 2 -> 1 (Original Size)
        x_dec = self.up5(x_dec)
        # Ensure output matches input spatial dimensions exactly
        if x_dec.size()[2:] != x.size()[2:]:
            x_dec = F.interpolate(
                x_dec, size=x.shape[2:], mode="bilinear", align_corners=True
            )
        x_dec = self.conv5(x_dec)

        # Final projection
        seg_logits = self.seg_head(x_dec)

        return cls_logits, seg_logits
