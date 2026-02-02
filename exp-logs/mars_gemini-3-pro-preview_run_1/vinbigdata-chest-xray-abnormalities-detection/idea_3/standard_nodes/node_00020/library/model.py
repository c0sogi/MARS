import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution used in BiFPN to reduce parameters.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.act = Swish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    Single BiFPN Layer implementing weighted feature fusion.
    Nodes: P3, P4, P5, P6, P7.
    """

    def __init__(self, channels):
        super().__init__()
        self.epsilon = 1e-4

        # Learnable weights for fusion (Fast Normalized Fusion)
        # P6_td: Inputs P6_in, P7_in
        self.w_p6_td = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.conv_p6_td = SeparableConvBlock(channels, channels)

        # P5_td: Inputs P5_in, P6_td
        self.w_p5_td = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.conv_p5_td = SeparableConvBlock(channels, channels)

        # P4_td: Inputs P4_in, P5_td
        self.w_p4_td = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.conv_p4_td = SeparableConvBlock(channels, channels)

        # P3_out: Inputs P3_in, P4_td
        self.w_p3_out = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.conv_p3_out = SeparableConvBlock(channels, channels)

        # P4_out: Inputs P4_in, P4_td, P3_out
        self.w_p4_out = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.conv_p4_out = SeparableConvBlock(channels, channels)

        # P5_out: Inputs P5_in, P5_td, P4_out
        self.w_p5_out = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.conv_p5_out = SeparableConvBlock(channels, channels)

        # P6_out: Inputs P6_in, P6_td, P5_out
        self.w_p6_out = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.conv_p6_out = SeparableConvBlock(channels, channels)

        # P7_out: Inputs P7_in, P6_out
        self.w_p7_out = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.conv_p7_out = SeparableConvBlock(channels, channels)

    def _normalize_weights(self, weights):
        w = torch.relu(weights)
        return w / (torch.sum(w, dim=0) + self.epsilon)

    def forward(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # --- Top-Down Pathway ---
        # P6_td
        w = self._normalize_weights(self.w_p6_td)
        p6_td = self.conv_p6_td(
            w[0] * p6_in + w[1] * F.interpolate(p7_in, scale_factor=2, mode="nearest")
        )

        # P5_td
        w = self._normalize_weights(self.w_p5_td)
        p5_td = self.conv_p5_td(
            w[0] * p5_in + w[1] * F.interpolate(p6_td, scale_factor=2, mode="nearest")
        )

        # P4_td
        w = self._normalize_weights(self.w_p4_td)
        p4_td = self.conv_p4_td(
            w[0] * p4_in + w[1] * F.interpolate(p5_td, scale_factor=2, mode="nearest")
        )

        # --- Bottom-Up Pathway ---
        # P3_out
        w = self._normalize_weights(self.w_p3_out)
        p3_out = self.conv_p3_out(
            w[0] * p3_in + w[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest")
        )

        # P4_out
        w = self._normalize_weights(self.w_p4_out)
        p4_out = self.conv_p4_out(
            w[0] * p4_in
            + w[1] * p4_td
            + w[2] * F.interpolate(p3_out, scale_factor=0.5, mode="nearest")
        )

        # P5_out
        w = self._normalize_weights(self.w_p5_out)
        p5_out = self.conv_p5_out(
            w[0] * p5_in
            + w[1] * p5_td
            + w[2] * F.interpolate(p4_out, scale_factor=0.5, mode="nearest")
        )

        # P6_out
        w = self._normalize_weights(self.w_p6_out)
        p6_out = self.conv_p6_out(
            w[0] * p6_in
            + w[1] * p6_td
            + w[2] * F.interpolate(p5_out, scale_factor=0.5, mode="nearest")
        )

        # P7_out
        w = self._normalize_weights(self.w_p7_out)
        p7_out = self.conv_p7_out(
            w[0] * p7_in
            + w[1] * F.interpolate(p6_out, scale_factor=0.5, mode="nearest")
        )

        return p3_out, p4_out, p5_out, p6_out, p7_out


class CoordinateInjector(nn.Module):
    """
    Appends normalized x, y coordinates (meshgrid) to the feature map.
    This allows the subsequent convolution layers (CoordinateConv) to learn spatial priors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # Create y coordinates: range [-1, 1] varying along height
        y_coords = (
            torch.linspace(-1, 1, H, device=x.device)
            .view(1, 1, H, 1)
            .repeat(B, 1, 1, W)
        )

        # Create x coordinates: range [-1, 1] varying along width
        x_coords = (
            torch.linspace(-1, 1, W, device=x.device)
            .view(1, 1, 1, W)
            .repeat(B, 1, H, 1)
        )

        # Concatenate along channel dimension
        return torch.cat([x, x_coords, y_coords], dim=1)


class SpatiallyAwareCenterNet(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()

        # 1. Backbone: EfficientNet-B3
        # We extract features at strides 4, 8, 16, 32
        # Indices: 1 (C2), 2 (C3), 3 (C4), 4 (C5)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        feature_channels = (
            self.backbone.feature_info.channels()
        )  # e.g., [32, 48, 136, 384]

        # 2. Neck: Stacked BiFPN
        self.fpn_channels = 128
        self.num_bifpn_layers = 3

        # Projections to bring backbone features to FPN channel size
        self.p3_proj = nn.Conv2d(feature_channels[1], self.fpn_channels, 1)  # stride 8
        self.p4_proj = nn.Conv2d(feature_channels[2], self.fpn_channels, 1)  # stride 16
        self.p5_proj = nn.Conv2d(feature_channels[3], self.fpn_channels, 1)  # stride 32

        # Generate P6, P7 from C5
        self.p6_conv = nn.Conv2d(
            feature_channels[3], self.fpn_channels, 3, stride=2, padding=1
        )
        self.p7_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, stride=2, padding=1),
        )

        # Stack BiFPN layers
        self.bifpn = nn.Sequential(
            *[BiFPNBlock(self.fpn_channels) for _ in range(self.num_bifpn_layers)]
        )

        # 3. Decoder / Upsampling
        # We use the refined P3 (stride 8) from BiFPN, upsample it to stride 4,
        # and fuse it with the projected C2 (stride 4) from backbone.
        self.c2_proj = nn.Conv2d(feature_channels[0], self.fpn_channels, 1)
        self.upsample_fuse = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            Swish(),
        )

        # 4. Coordinate Injection
        self.coord_injector = CoordinateInjector()
        # Input channels to heads = FPN channels + 2 (x, y)
        head_in_channels = self.fpn_channels + 2

        # 5. Heads
        # Heatmap Head (Classification)
        self.hm_head = nn.Sequential(
            nn.Conv2d(head_in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

        # Size Head (Width, Height)
        self.wh_head = nn.Sequential(
            nn.Conv2d(head_in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # Offset Head (Refinement)
        self.reg_head = nn.Sequential(
            nn.Conv2d(head_in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # Global Classification Head (Finding vs No Finding)
        # Operates on P7 (deepest layer, stride 128)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fpn_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Output logit for "No Finding" (Class 14)
        )

        # Weight Initialization for Heatmap Bias (Focal Loss stability)
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # 1. Backbone Extraction
        feats = self.backbone(x)
        c2, c3, c4, c5 = feats  # strides 4, 8, 16, 32

        # 2. Prepare BiFPN Inputs
        p3 = self.p3_proj(c3)
        p4 = self.p4_proj(c4)
        p5 = self.p5_proj(c5)
        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(p6)

        # 3. BiFPN Forward
        features = (p3, p4, p5, p6, p7)
        for layer in self.bifpn:
            features = layer(features)

        p3_out, _, _, _, p7_out = features

        # 4. Upsample and Fuse (Decoder)
        # Upsample P3 (stride 8) -> stride 4
        p3_up = F.interpolate(p3_out, scale_factor=2, mode="nearest")
        # Project C2 (stride 4)
        c2_proj = self.c2_proj(c2)
        # Fuse
        fused_map = p3_up + c2_proj
        fused_map = self.upsample_fuse(fused_map)

        # 5. Coordinate Injection
        fused_map_coords = self.coord_injector(fused_map)

        # 6. Heads
        hm = self.hm_head(fused_map_coords)
        wh = self.wh_head(fused_map_coords)
        reg = self.reg_head(fused_map_coords)

        # Global Head (Probability of "No Finding")
        global_logit = self.global_head(p7_out)

        return {
            "hm": torch.sigmoid(hm),
            "wh": wh,
            "reg": reg,
            "global_no_finding": global_logit,
        }
