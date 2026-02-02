import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class CoordConv(nn.Module):
    """
    Coordinate Convolution Layer.
    Appends normalized x and y coordinates to the input feature map.
    """

    def __init__(self):
        super(CoordConv, self).__init__()

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Create y and x grids
        y_range = torch.linspace(-1, 1, height, device=x.device)
        x_range = torch.linspace(-1, 1, width, device=x.device)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        # Expand to batch size: (B, 1, H, W)
        y_grid = y_grid.expand(batch_size, 1, height, width)
        x_grid = x_grid.expand(batch_size, 1, height, width)

        # Concatenate coordinate channels to the input features
        out = torch.cat([x, x_grid, y_grid], dim=1)
        return out


class ConvBlock(nn.Module):
    """
    Standard Conv-BN-ReLU block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiFPNBlock(nn.Module):
    """
    Simplified FPN/BiFPN block to fuse multi-scale features.
    Projects features to common channels and fuses top-down to produce a stride-4 output.
    """

    def __init__(self, in_channels_list, out_channels):
        super(BiFPNBlock, self).__init__()

        # Lateral layers to project backbone features to BIFPN_CHANNELS
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels_list]
        )

        # Convolutions after fusion to smooth aliasing
        self.fpn_convs = nn.ModuleList(
            [
                ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, inputs):
        # inputs: [C2, C3, C4, C5] from EfficientNet (strides 4, 8, 16, 32)

        # 1. Lateral projections
        projections = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        p2, p3, p4, p5 = projections

        # 2. Top-down pathway (Fusing deeper features into shallower ones)
        # P5 -> P4
        p4_fused = p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_out = self.fpn_convs[2](p4_fused)

        # P4 -> P3
        p3_fused = p3 + F.interpolate(p4_out, size=p3.shape[2:], mode="nearest")
        p3_out = self.fpn_convs[1](p3_fused)

        # P3 -> P2 (Target Stride 4)
        p2_fused = p2 + F.interpolate(p3_out, size=p2.shape[2:], mode="nearest")
        p2_out = self.fpn_convs[0](p2_fused)

        # We return the highest resolution feature map (Stride 4) for CenterNet
        return p2_out


class AnatomicalCenterNet(nn.Module):
    def __init__(self):
        super(AnatomicalCenterNet, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # features_only=True returns a list of feature maps
        # out_indices=(1, 2, 3, 4) corresponds to strides 4, 8, 16, 32 for EfficientNet
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get channel counts for the selected indices
        # EfficientNet-B0 channels: [24, 40, 112, 320] for indices 1, 2, 3, 4
        feature_info = self.backbone.feature_info.channels()

        # 2. Neck: Feature Fusion
        self.neck = BiFPNBlock(feature_info, Config.BIFPN_CHANNELS)

        # 3. CoordConv
        self.use_coord_conv = Config.USE_COORD_CONV
        self.coord_conv = CoordConv()

        # Input channels to heads
        # If CoordConv is used, we add 2 channels (x, y)
        head_in_channels = Config.BIFPN_CHANNELS + (2 if self.use_coord_conv else 0)

        # 4. Detection Heads
        # Heatmap Head (Classes)
        self.hm_head = nn.Sequential(
            ConvBlock(head_in_channels, head_in_channels),
            nn.Conv2d(head_in_channels, 14, kernel_size=1),  # 14 findings
            # Sigmoid is applied in loss or inference, but usually raw logits for Focal Loss
        )

        # Size Head (Width, Height)
        self.wh_head = nn.Sequential(
            ConvBlock(head_in_channels, head_in_channels),
            nn.Conv2d(head_in_channels, 2, kernel_size=1),
        )

        # Offset Head (x, y local offset)
        self.reg_head = nn.Sequential(
            ConvBlock(head_in_channels, head_in_channels),
            nn.Conv2d(head_in_channels, 2, kernel_size=1),
        )

        # 5. Global Classification Head (Auxiliary)
        # Operates on the deepest feature map (C5, stride 32, 320 channels)
        c5_channels = feature_info[-1]
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # Binary: Finding vs No Finding
            # Sigmoid applied in loss/inference
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize heads
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Special initialization for heatmap head (focal loss prior)
        # bias = -log((1 - pi) / pi) with pi = 0.01
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone extraction
        # features: [C2, C3, C4, C5]
        features = self.backbone(x)

        # Neck: Fuse features to Stride 4
        neck_out = self.neck(features)

        # CoordConv: Add spatial awareness
        if self.use_coord_conv:
            neck_out = self.coord_conv(neck_out)

        # Detection Heads
        hm = self.hm_head(neck_out)
        wh = self.wh_head(neck_out)
        reg = self.reg_head(neck_out)

        # Global Classification Head
        # Use the deepest feature map (C5) for global context
        global_feat = features[-1]
        global_out = self.global_head(global_feat)

        # Apply sigmoid to heatmap and global head for output consistency
        # Note: During training with Focal Loss, we might use logits,
        # but standard CenterNet implementation often returns sigmoid for hm.
        # We will return sigmoid for hm and global here for simplicity in inference/loss.
        hm = torch.sigmoid(hm)
        global_out = torch.sigmoid(global_out)

        return {"hm": hm, "wh": wh, "reg": reg, "global_label": global_out}
