import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.modules import LayerNorm, ConvNeXtBlock, SCSEModule, ASPP
from library.config import Config


class SingleStreamUNet(nn.Module):
    """
    Single-Stream Large-Kernel U-Net (Cite solution_lesson_node_00047).

    Architecture:
    - Single ConvNeXt-Tiny encoder processing stacked 6-channel input (Early Fusion).
    - ASPP Bottleneck.
    - Deep Sequential Decoder extending to Stride 1 (Cite solution_lesson_node_00039).
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_chans=None,
        num_classes=1,
    ):
        super().__init__()
        if in_chans is None:
            in_chans = Config.IN_CHANNELS

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
        )

        # Extract channel dimensions
        # ConvNeXt-Tiny: [96, 192, 384, 768] for strides [4, 8, 16, 32]
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

        # Stage 0: Stride 4 -> 2 (Learned Upsampling)
        # Input: Stage 1 output (dims[0])
        # No skip connection (Encoder stem is stride 4)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # We process at dim=dims[0] (96)
        self.block0 = ConvNeXtBlock(dims[0])

        # Stage Final: Stride 2 -> 1 (Learned Upsampling)
        # Input: Stage 0 output (dims[0])
        self.up_final = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.block_final = ConvNeXtBlock(dims[0])

        # Final Projection
        self.final_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Encoder Pass
        # ---------------------------------------------------------------------
        # feats indices: 0->s4, 1->s8, 2->s16, 3->s32
        feats = self.encoder(x)

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        x = self.aspp(feats[3])

        # ---------------------------------------------------------------------
        # Decoder Pass
        # ---------------------------------------------------------------------

        # Decoder Stage 3 (32 -> 16)
        x = self.up3(x)
        if x.shape[2:] != feats[2].shape[2:]:
            x = F.interpolate(
                x, size=feats[2].shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, feats[2]], dim=1)
        x = self.scse3(x)
        x = self.reduce3(x)
        x = self.block3(x)

        # Decoder Stage 2 (16 -> 8)
        x = self.up2(x)
        if x.shape[2:] != feats[1].shape[2:]:
            x = F.interpolate(
                x, size=feats[1].shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, feats[1]], dim=1)
        x = self.scse2(x)
        x = self.reduce2(x)
        x = self.block2(x)

        # Decoder Stage 1 (8 -> 4)
        x = self.up1(x)
        if x.shape[2:] != feats[0].shape[2:]:
            x = F.interpolate(
                x, size=feats[0].shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, feats[0]], dim=1)
        x = self.scse1(x)
        x = self.reduce1(x)
        x = self.block1(x)

        # Decoder Stage 0 (4 -> 2)
        x = self.up0(x)
        x = self.block0(x)

        # Decoder Stage Final (2 -> 1)
        x = self.up_final(x)
        x = self.block_final(x)

        # Final Output
        logits = self.final_head(x)

        return logits
