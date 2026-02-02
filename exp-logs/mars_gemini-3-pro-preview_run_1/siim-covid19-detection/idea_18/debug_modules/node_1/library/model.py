import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DualPoolingHead(nn.Module):
    """
    Dual Pooling Head: Concatenates Global Average Pooling and Global Std Deviation Pooling.
    Used to capture texture information (variance) which is crucial for 'Atypical' vs 'Indeterminate'.
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0.0):
        super().__init__()
        # Input will be concatenated mean and std, so 2 * in_channels
        self.fc = nn.Linear(in_channels * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (B, C, H, W)

        # Global Average Pooling
        x_mean = torch.mean(x, dim=(2, 3))

        # Global Standard Deviation Pooling
        # unbiased=False matches numpy's default std (population std) often used in these contexts.
        # It is generally more stable than sample std when spatial dims are small.
        x_std = torch.std(x, dim=(2, 3), unbiased=False)

        # Concatenate
        x_cat = torch.cat([x_mean, x_std], dim=1)

        # Dropout and Linear
        x_cat = self.dropout(x_cat)
        return self.fc(x_cat)


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block with Bilinear Upsampling and Skip Connection concatenation.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
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

    def forward(self, x, skip):
        # Upsample input
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle potential padding issues if dimensions are not perfect powers of 2
        # (Though 512x512 with ResNet34 is safe, this ensures robustness)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class StochasticDepthResNet34UNet(nn.Module):
    """
    ResNet34 U-Net with Stochastic Depth (DropPath) and Dual-Pooling Classification Head.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (Backbone)
        # Using timm to load ResNet34 with Stochastic Depth
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            drop_path_rate=Config.DROP_PATH_RATE,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts from the backbone
        # For ResNet34: [64, 64, 128, 256, 512]
        # Corresponding to strides: [2, 4, 8, 16, 32]
        enc_channels = self.encoder.feature_info.channels()

        # 2. Classification Head
        # Attached to the final feature map (stride 32, 512 channels)
        self.cls_head = DualPoolingHead(
            in_channels=enc_channels[-1],
            num_classes=Config.NUM_STUDY_CLASSES,
            dropout_rate=Config.DROPOUT_RATE,
        )

        # 3. Segmentation Decoder
        # We construct the decoder path using the skip connections from the encoder
        decoder_channels = Config.DECODER_CHANNELS  # [256, 128, 64, 32, 16]

        # Block 1: Input (512) + Skip (256) -> Out (256)
        self.dec1 = DecoderBlock(enc_channels[4], enc_channels[3], decoder_channels[0])

        # Block 2: Input (256) + Skip (128) -> Out (128)
        self.dec2 = DecoderBlock(
            decoder_channels[0], enc_channels[2], decoder_channels[1]
        )

        # Block 3: Input (128) + Skip (64) -> Out (64)
        self.dec3 = DecoderBlock(
            decoder_channels[1], enc_channels[1], decoder_channels[2]
        )

        # Block 4: Input (64) + Skip (64) -> Out (32)
        self.dec4 = DecoderBlock(
            decoder_channels[2], enc_channels[0], decoder_channels[3]
        )

        # Final Block: Upsample (32) -> Out (16) -> Conv1x1 (1)
        # This brings stride 2 (256x256) up to stride 1 (512x512)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                decoder_channels[3], decoder_channels[4], kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[4], 1, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder ---
        # Returns a list of features
        features = self.encoder(x)
        # features[0]: Stride 2 (64 ch)
        # features[1]: Stride 4 (64 ch)
        # features[2]: Stride 8 (128 ch)
        # features[3]: Stride 16 (256 ch)
        # features[4]: Stride 32 (512 ch)

        # --- Classification Branch ---
        # Use the deepest feature map
        cls_logits = self.cls_head(features[4])

        # --- Segmentation Branch ---
        # Start from the bottom
        x = features[4]

        # Decode
        x = self.dec1(x, features[3])
        x = self.dec2(x, features[2])
        x = self.dec3(x, features[1])
        x = self.dec4(x, features[0])

        # Final upsample to original resolution
        seg_logits = self.final_upsample(x)

        return cls_logits, seg_logits
