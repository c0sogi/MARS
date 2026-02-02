import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.modules import LayerNorm, ConvNeXtBlock, SCSEModule, ASPP
from library.config import Config


class DualStreamUNet(nn.Module):
    """
    Dual-Stream Large-Kernel U-Net with Cross-Modal Fusion.

    Architecture:
    - Two parallel ConvNeXt-Tiny encoders (Stream A: Static, Stream B: Dynamic).
    - Cross-Modal Feature Fusion at each stage via concatenation and 1x1 convolution.
    - ASPP Bottleneck.
    - Sequential U-Net Decoder with ConvNeXt Blocks and SCSE Attention.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_chans_a=Config.IN_CHANNELS_STREAM_A,
        in_chans_b=Config.IN_CHANNELS_STREAM_B,
        num_classes=1,
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # Encoders
        # ---------------------------------------------------------------------
        # Stream A: Static/Spectral (Ash Composite)
        self.encoder_a = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans_a,
        )

        # Stream B: Dynamic/Temporal (Band Differences)
        self.encoder_b = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans_b,
        )

        # Extract channel dimensions from the backbone
        # ConvNeXt-Tiny typically returns features with channels: [96, 192, 384, 768]
        # corresponding to strides: [4, 8, 16, 32]
        feature_info = self.encoder_a.feature_info
        dims = feature_info.channels()

        # ---------------------------------------------------------------------
        # Cross-Modal Fusion Modules
        # ---------------------------------------------------------------------
        # Fuses features from both streams at each stage.
        # Concatenation (dim*2) -> 1x1 Conv -> dim
        self.fusion_0 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
        )
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(inplace=True),
        )
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, dims[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        # ASPP applied to the deepest fused features (Stride 32)
        self.aspp = ASPP(dims[3], dims[3])

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        # We use a sequential refinement approach with ConvNeXt Blocks.

        # Stage 3: Stride 32 -> 16
        # Input: ASPP output (dims[3])
        # Skip: Fused Stage 2 (dims[2])
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse3 = SCSEModule(dims[3] + dims[2])
        self.reduce3 = nn.Conv2d(dims[3] + dims[2], dims[2], kernel_size=1)
        self.block3 = ConvNeXtBlock(dims[2])

        # Stage 2: Stride 16 -> 8
        # Input: Stage 3 output (dims[2])
        # Skip: Fused Stage 1 (dims[1])
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse2 = SCSEModule(dims[2] + dims[1])
        self.reduce2 = nn.Conv2d(dims[2] + dims[1], dims[1], kernel_size=1)
        self.block2 = ConvNeXtBlock(dims[1])

        # Stage 1: Stride 8 -> 4
        # Input: Stage 2 output (dims[1])
        # Skip: Fused Stage 0 (dims[0])
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse1 = SCSEModule(dims[1] + dims[0])
        self.reduce1 = nn.Conv2d(dims[1] + dims[0], dims[0], kernel_size=1)
        self.block1 = ConvNeXtBlock(dims[0])

        # Final Projection: Stride 4 -> 1
        # ConvNeXt stem is stride 4, so we need 4x upsampling to reach original resolution.
        self.final_up = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.final_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C_total, H, W).
               Expected to contain stacked Static (Stream A) and Dynamic (Stream B) channels.
        """
        # Split input into Static and Dynamic streams
        # Stream A: First 3 channels (Ash)
        # Stream B: Next 3 channels (Diff)
        x_static = x[:, :3, :, :]
        x_dynamic = x[:, 3:, :, :]

        # ---------------------------------------------------------------------
        # Encoder Pass
        # ---------------------------------------------------------------------
        # feats indices: 0->s4, 1->s8, 2->s16, 3->s32
        feats_a = self.encoder_a(x_static)
        feats_b = self.encoder_b(x_dynamic)

        # ---------------------------------------------------------------------
        # Feature Fusion
        # ---------------------------------------------------------------------
        f0 = self.fusion_0(torch.cat([feats_a[0], feats_b[0]], dim=1))
        f1 = self.fusion_1(torch.cat([feats_a[1], feats_b[1]], dim=1))
        f2 = self.fusion_2(torch.cat([feats_a[2], feats_b[2]], dim=1))
        f3 = self.fusion_3(torch.cat([feats_a[3], feats_b[3]], dim=1))

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        x = self.aspp(f3)

        # ---------------------------------------------------------------------
        # Decoder Pass
        # ---------------------------------------------------------------------

        # Decoder Stage 3 (32 -> 16)
        x = self.up3(x)
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(
                x, size=f2.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, f2], dim=1)
        x = self.scse3(x)
        x = self.reduce3(x)
        x = self.block3(x)

        # Decoder Stage 2 (16 -> 8)
        x = self.up2(x)
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(
                x, size=f1.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, f1], dim=1)
        x = self.scse2(x)
        x = self.reduce2(x)
        x = self.block2(x)

        # Decoder Stage 1 (8 -> 4)
        x = self.up1(x)
        if x.shape[2:] != f0.shape[2:]:
            x = F.interpolate(
                x, size=f0.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, f0], dim=1)
        x = self.scse1(x)
        x = self.reduce1(x)
        x = self.block1(x)

        # Final Output (4 -> 1)
        x = self.final_up(x)
        logits = self.final_head(x)

        return logits
