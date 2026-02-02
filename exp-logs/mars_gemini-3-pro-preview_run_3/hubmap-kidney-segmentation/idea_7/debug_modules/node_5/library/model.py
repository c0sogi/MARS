import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle potential shape mismatch due to padding/cropping in encoder
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        classes,
        decoder_channels,
        deep_supervision=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Load encoder from timm
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=(encoder_weights is not None),
            out_indices=(0, 1, 2, 3),
        )

        # Get encoder channel counts
        # e.g., convnext_tiny: [96, 192, 384, 768] for strides [4, 8, 16, 32]
        enc_channels = self.encoder.feature_info.channels()

        # Ensure we have enough decoder channels config
        assert len(decoder_channels) >= 5, "Decoder requires at least 5 channel configs"

        self.decoder_blocks = nn.ModuleList()

        # Block 1: stride 32 -> 16
        # Input: enc_channels[3] (768), Skip: enc_channels[2] (384)
        self.decoder_blocks.append(
            DecoderBlock(enc_channels[3], enc_channels[2], decoder_channels[0])
        )

        # Block 2: stride 16 -> 8
        # Input: decoder_channels[0], Skip: enc_channels[1] (192)
        self.decoder_blocks.append(
            DecoderBlock(decoder_channels[0], enc_channels[1], decoder_channels[1])
        )

        # Block 3: stride 8 -> 4
        # Input: decoder_channels[1], Skip: enc_channels[0] (96)
        self.decoder_blocks.append(
            DecoderBlock(decoder_channels[1], enc_channels[0], decoder_channels[2])
        )

        # Block 4: stride 4 -> 2
        # Input: decoder_channels[2], Skip: None (0)
        self.decoder_blocks.append(
            DecoderBlock(decoder_channels[2], 0, decoder_channels[3])
        )

        # Block 5: stride 2 -> 1
        # Input: decoder_channels[3], Skip: None (0)
        self.decoder_blocks.append(
            DecoderBlock(decoder_channels[3], 0, decoder_channels[4])
        )

        # Final Convolution
        self.final_conv = nn.Conv2d(decoder_channels[4], classes, kernel_size=1)

        # Deep Supervision Heads
        if self.deep_supervision:
            # Create heads for intermediate layers (Block 3, Block 2, Block 1)
            # This matches the 4 loss weights when combined with final_conv
            self.deep_heads = nn.ModuleList(
                [
                    nn.Conv2d(decoder_channels[3], classes, kernel_size=1),
                    nn.Conv2d(decoder_channels[2], classes, kernel_size=1),
                    nn.Conv2d(decoder_channels[1], classes, kernel_size=1),
                ]
            )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features list: [stride4, stride8, stride16, stride32]

        # Decoder
        x = features[3]  # Start with deepest feature

        x = self.decoder_blocks[0](x, features[2])  # 32 -> 16
        x_b1 = self.decoder_blocks[1](x, features[1])  # 16 -> 8
        x_b2 = self.decoder_blocks[2](x_b1, features[0])  # 8 -> 4
        x_b3 = self.decoder_blocks[3](x_b2)  # 4 -> 2
        x_final = self.decoder_blocks[4](x_b3)  # 2 -> 1

        logits = self.final_conv(x_final)

        if self.deep_supervision:
            outputs = [logits]
            target_size = logits.shape[-2:]
            intermediates = [x_b3, x_b2, x_b1]

            for i, head in enumerate(self.deep_heads):
                if i < len(intermediates):
                    int_logits = head(intermediates[i])
                    # Upsample to match final output size
                    int_logits = F.interpolate(
                        int_logits,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    outputs.append(int_logits)

            return outputs

        return logits


def build_model():
    """
    Constructs the custom U-Net model with a timm backbone.
    """
    model = Unet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        classes=Config.CLASSES,
        decoder_channels=Config.DECODER_CHANNELS,
        deep_supervision=Config.DEEP_SUPERVISION,
    )

    return model
