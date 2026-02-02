import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DepthProjector(nn.Module):
    """
    Non-linear MLP to project scalar depth into a dense embedding.
    Structure: Linear(1->16) -> ReLU -> Linear(16->32).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Performs: Conv1x1 (reduce) -> TransposeConv3x3 (upsample) -> Conv1x1 (expand).
    Internal width is in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Cite solution_lesson_node_00023: Calculate internal width from in_channels
        mid_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv to reduce channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv to upsample
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv to expand/match output channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SaltLinkNet(nn.Module):
    """
    ResNet34-LinkNet with Explicit Depth Injection.
    Cite solution_lesson_node_00032: Explicit feature injection is superior to distillation.
    """

    def __init__(self):
        super().__init__()

        # Encoder: ResNet34
        # Cite solution_lesson_node_00002: Use pretrained ResNet34
        # in_chans=1 automatically sums pretrained RGB weights to 1 channel
        self.encoder = timm.create_model("resnet34", pretrained=True, in_chans=1)

        # Depth Injection
        # Cite solution_lesson_node_00029: Non-linear MLP for depth embedding
        self.depth_projector = DepthProjector()

        # ResNet34 layer4 out is 512. Depth embedding is 32.
        # Cite solution_lesson_node_00001: Inject at bottleneck
        # Cite solution_lesson_node_00009: Concatenate small embedding
        neck_channels = 512 + 32

        # Decoder Blocks
        # ResNet34 channels: Layer4=512, Layer3=256, Layer2=128, Layer1=64, Stem(Conv1)=64

        # Dec4: Takes Bottleneck -> Matches Layer3 (256)
        # Cite solution_lesson_node_00019: No redundant bottleneck layer
        self.dec4 = DecoderBlock(neck_channels, 256)

        # Dec3: Takes Dec4+Layer3 -> Matches Layer2 (128)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: Takes Dec3+Layer2 -> Matches Layer1 (64)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: Takes Dec2+Layer1 -> Matches Stem/Conv1 (64)
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsampling Block: Matches Input Resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Prediction Head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth):
        # --- Encoder ---
        # Stem
        x = self.encoder.conv1(x)  # (B, 64, H/2, W/2) -> ~51x51
        x = self.encoder.bn1(x)
        c1 = self.encoder.act1(x)  # Skip connection 1 (Stem)

        x = self.encoder.maxpool(c1)  # (B, 64, H/4, W/4) -> ~26x26

        # Layers
        l1 = self.encoder.layer1(x)  # (B, 64, 26, 26) -> Skip connection 2
        l2 = self.encoder.layer2(l1)  # (B, 128, 13, 13) -> Skip connection 3
        l3 = self.encoder.layer3(l2)  # (B, 256, 7, 7)   -> Skip connection 4
        l4 = self.encoder.layer4(l3)  # (B, 512, 4, 4)   -> Bottleneck base

        # --- Bottleneck ---
        # Cite solution_lesson_node_00001: Depth injection at bottleneck
        if depth is None:
            raise ValueError("Depth must be provided.")

        # Project depth
        d = self.depth_projector(depth)  # (B, 32)

        # Expand spatially to match l4
        d = d.view(d.size(0), 32, 1, 1).expand(-1, -1, l4.size(2), l4.size(3))

        # Concatenate
        neck = torch.cat([l4, d], dim=1)  # (B, 544, 4, 4)

        # --- Decoder ---
        # Helper for additive skip connection with resizing
        def add_skip(dec, skip):
            if dec.size()[2:] != skip.size()[2:]:
                dec = F.interpolate(
                    dec, size=skip.size()[2:], mode="bilinear", align_corners=False
                )
            return dec + skip

        # Block 4
        d4 = self.dec4(neck)
        d4 = add_skip(d4, l3)

        # Block 3
        d3 = self.dec3(d4)
        d3 = add_skip(d3, l2)

        # Block 2
        d2 = self.dec2(d3)
        d2 = add_skip(d2, l1)

        # Block 1
        d1 = self.dec1(d2)
        d1 = add_skip(d1, c1)

        # --- Final Upsample ---
        out = self.final_up(d1)

        # Ensure final size matches 101x101 (original input)
        if out.size()[2:] != (101, 101):
            out = F.interpolate(
                out, size=(101, 101), mode="bilinear", align_corners=False
            )

        logits = self.head(out)

        return logits
