import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.modules import LayerNorm, ConvNeXtBlock, SCSEModule, ASPP
from library.config import Config


class ContextEnhancedUNet(nn.Module):
    """
    Single-Stream Large-Kernel U-Net (Idea 11).
    Cite solution_lesson_node_00047: Prefer Early Fusion (Channel Stacking) over Multi-Stream.
    Cite solution_lesson_node_00036: Large-Kernel Convolutions (ConvNeXt) for Elongated Features.
    Cite solution_lesson_node_00046: Sequential Refinement over Pyramid Fusion.

    Architecture:
    - Single ConvNeXt-Tiny encoder (6 input channels).
    - ASPP Bottleneck.
    - Sequential U-Net Decoder with ConvNeXt Blocks and SCSE Attention.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_chans=None,
        num_classes=1,
    ):
        super().__init__()
        in_chans = in_chans if in_chans is not None else Config.IN_CHANNELS

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
        )

        # Extract channel dimensions from the backbone
        # ConvNeXt-Tiny typically returns features with channels: [96, 192, 384, 768]
        # corresponding to strides: [4, 8, 16, 32]
        feature_info = self.encoder.feature_info
        dims = feature_info.channels()

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        # ASPP applied to the deepest features (Stride 32)
        self.aspp = ASPP(dims[3], dims[3])

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        # We use a sequential refinement approach with ConvNeXt Blocks.

        # Stage 3: Stride 32 -> 16
        # Input: ASPP output (dims[3])
        # Skip: Encoder Stage 2 (dims[2])
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse3 = SCSEModule(dims[3] + dims[2])
        self.reduce3 = nn.Conv2d(dims[3] + dims[2], dims[2], kernel_size=1)
        self.block3 = ConvNeXtBlock(dims[2])

        # Stage 2: Stride 16 -> 8
        # Input: Stage 3 output (dims[2])
        # Skip: Encoder Stage 1 (dims[1])
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse2 = SCSEModule(dims[2] + dims[1])
        self.reduce2 = nn.Conv2d(dims[2] + dims[1], dims[1], kernel_size=1)
        self.block2 = ConvNeXtBlock(dims[1])

        # Stage 1: Stride 8 -> 4
        # Input: Stage 2 output (dims[1])
        # Skip: Encoder Stage 0 (dims[0])
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.scse1 = SCSEModule(dims[1] + dims[0])
        self.reduce1 = nn.Conv2d(dims[1] + dims[0], dims[0], kernel_size=1)
        self.block1 = ConvNeXtBlock(dims[0])

        # Final Projection: Stride 4 -> 1
        # ConvNeXt stem is stride 4, so we need 4x upsampling to reach original resolution.
        # Cite solution_lesson_node_00039: We should ideally use learned upsampling to s1,
        # but Idea 11 (0.5911) used this structure successfully.
        # Given the constraints and success of Idea 11, we replicate it.
        self.final_up = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.final_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 6, H, W).
        """
        # ---------------------------------------------------------------------
        # Encoder Pass
        # ---------------------------------------------------------------------
        # feats indices: 0->s4, 1->s8, 2->s16, 3->s32
        feats = self.encoder(x)
        f0, f1, f2, f3 = feats[0], feats[1], feats[2], feats[3]

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
