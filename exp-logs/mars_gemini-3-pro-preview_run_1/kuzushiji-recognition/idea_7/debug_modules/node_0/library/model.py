import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from library.config import Config


class DeformableConvBlock(nn.Module):
    """
    Applies a Deformable Convolution v2.
    Generates offsets and masks dynamically from the input features.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = 1

        # Offset and Mask generator
        # We need 2 offsets (x, y) per kernel element and 1 mask per kernel element
        # Total channels: 3 * kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Initialize offset/mask weights to 0 so training starts as a standard convolution
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

        self.dcn = torchvision.ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Generate offsets and masks
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)  # Masks must be in [0, 1]

        # Apply Deformable Convolution
        x = self.dcn(x, offset, mask)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvNeXtCenterNet(nn.Module):
    """
    CenterNet architecture with ConvNeXt-Tiny backbone, FPN neck,
    and Deformable Convolution heads.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # 1. Backbone: ConvNeXt Tiny
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, features_only=True
        )

        # Get channel counts for stages. Typically [96, 192, 384, 768] for strides [4, 8, 16, 32]
        feature_dims = self.backbone.feature_info.channels()
        fpn_dim = 256

        # 2. FPN Lateral Connections (1x1 Convs)
        self.lat_c5 = nn.Conv2d(feature_dims[3], fpn_dim, 1)
        self.lat_c4 = nn.Conv2d(feature_dims[2], fpn_dim, 1)
        self.lat_c3 = nn.Conv2d(feature_dims[1], fpn_dim, 1)
        self.lat_c2 = nn.Conv2d(feature_dims[0], fpn_dim, 1)

        # 3. Heads
        # All heads operate on the fused FPN feature map (Stride 4)

        # Heatmap Head: Class Agnostic (1 Channel)
        # Predicts "objectness" probability
        self.hm_head = nn.Sequential(
            DeformableConvBlock(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 1, 1),
            nn.Sigmoid(),
        )

        # Regression Head: 4 Channels
        # Predicts [offset_x, offset_y, width, height]
        self.reg_head = nn.Sequential(
            DeformableConvBlock(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 4, 1)
        )

        # Classification Head: NumClasses Channels
        # Predicts class logits at the center point
        self.cls_head = nn.Sequential(
            DeformableConvBlock(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Standard initialization for heads
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize heatmap bias to -2.19 (approx sigmoid(-2.19) = 0.1)
        # This prevents a massive loss at the start of training (Focal Loss prior)
        self.hm_head[-2].bias.data.fill_(-2.19)

    def forward(self, x):
        # --- Backbone ---
        features = self.backbone(x)
        # ConvNeXt features: [Stride 4, Stride 8, Stride 16, Stride 32]
        c2, c3, c4, c5 = features[0], features[1], features[2], features[3]

        # --- FPN (Feature Pyramid Network) ---
        # Top-down pathway with lateral connections

        # P5: Stride 32 -> Stride 32
        p5 = self.lat_c5(c5)

        # P4: Stride 16 + Upsampled P5
        p4 = self.lat_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")

        # P3: Stride 8 + Upsampled P4
        p3 = self.lat_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        # P2: Stride 4 + Upsampled P3
        p2 = self.lat_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # We use P2 (Stride 4) as the common feature map for detection

        # --- Heads ---
        hm = self.hm_head(p2)  # (B, 1, H/4, W/4)
        reg_wh = self.reg_head(p2)  # (B, 4, H/4, W/4)
        cls_logits = self.cls_head(p2)  # (B, NumClasses, H/4, W/4)

        return {"hm": hm, "reg_wh": reg_wh, "cls_logits": cls_logits}
