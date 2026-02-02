import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution used in BiFPN.
    """

    def __init__(self, in_channels, out_channels, norm=True, activation=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = Swish() if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network Block.
    Fuses features from P3 to P7.
    """

    def __init__(self, channels):
        super().__init__()
        self.epsilon = 1e-4

        # Learnable Weights for Fusion
        # Top-down path
        self.w_p6_td = nn.Parameter(torch.ones(2))
        self.w_p5_td = nn.Parameter(torch.ones(2))
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))  # P3 is leaf in TD, root in BU

        # Bottom-up path
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(3))
        self.w_p6_out = nn.Parameter(torch.ones(3))
        self.w_p7_out = nn.Parameter(torch.ones(2))

        # Fusion Convolutions
        self.conv_p6_td = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p5_td = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p4_td = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p3_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p4_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p5_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p6_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv_p7_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )

    def forward(self, p3, p4, p5, p6, p7):
        # --- Top-Down Path ---
        # P7 -> P6
        w_p6 = torch.relu(self.w_p6_td)
        w_p6 = w_p6 / (torch.sum(w_p6) + self.epsilon)
        p6_td = self.conv_p6_td(
            w_p6[0] * p6 + w_p6[1] * F.interpolate(p7, scale_factor=2, mode="nearest")
        )

        # P6 -> P5
        w_p5 = torch.relu(self.w_p5_td)
        w_p5 = w_p5 / (torch.sum(w_p5) + self.epsilon)
        p5_td = self.conv_p5_td(
            w_p5[0] * p5
            + w_p5[1] * F.interpolate(p6_td, scale_factor=2, mode="nearest")
        )

        # P5 -> P4
        w_p4 = torch.relu(self.w_p4_td)
        w_p4 = w_p4 / (torch.sum(w_p4) + self.epsilon)
        p4_td = self.conv_p4_td(
            w_p4[0] * p4
            + w_p4[1] * F.interpolate(p5_td, scale_factor=2, mode="nearest")
        )

        # P4 -> P3 (Output P3)
        w_p3 = torch.relu(self.w_p3_out)
        w_p3 = w_p3 / (torch.sum(w_p3) + self.epsilon)
        p3_out = self.conv_p3_out(
            w_p3[0] * p3
            + w_p3[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest")
        )

        # --- Bottom-Up Path ---
        # P3 -> P4
        w_p4_up = torch.relu(self.w_p4_out)
        w_p4_up = w_p4_up / (torch.sum(w_p4_up) + self.epsilon)
        p4_out = self.conv_p4_out(
            w_p4_up[0] * p4
            + w_p4_up[1] * p4_td
            + w_p4_up[2] * F.max_pool2d(p3_out, kernel_size=3, stride=2, padding=1)
        )

        # P4 -> P5
        w_p5_up = torch.relu(self.w_p5_out)
        w_p5_up = w_p5_up / (torch.sum(w_p5_up) + self.epsilon)
        p5_out = self.conv_p5_out(
            w_p5_up[0] * p5
            + w_p5_up[1] * p5_td
            + w_p5_up[2] * F.max_pool2d(p4_out, kernel_size=3, stride=2, padding=1)
        )

        # P5 -> P6
        w_p6_up = torch.relu(self.w_p6_out)
        w_p6_up = w_p6_up / (torch.sum(w_p6_up) + self.epsilon)
        p6_out = self.conv_p6_out(
            w_p6_up[0] * p6
            + w_p6_up[1] * p6_td
            + w_p6_up[2] * F.max_pool2d(p5_out, kernel_size=3, stride=2, padding=1)
        )

        # P6 -> P7
        w_p7_up = torch.relu(self.w_p7_out)
        w_p7_up = w_p7_up / (torch.sum(w_p7_up) + self.epsilon)
        p7_out = self.conv_p7_out(
            w_p7_up[0] * p7
            + w_p7_up[1] * F.max_pool2d(p6_out, kernel_size=3, stride=2, padding=1)
        )

        return p3_out, p4_out, p5_out, p6_out, p7_out


class CoordConv(nn.Module):
    """
    Appends normalized x, y coordinates to the feature map.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Normalized coordinates [-1, 1]
        x_range = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_range = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        x_grid = x_grid.expand(b, 1, -1, -1)
        y_grid = y_grid.expand(b, 1, -1, -1)

        out = torch.cat([x, x_grid, y_grid], dim=1)
        return self.conv(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) to focus on relevant spatial regions.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * torch.sigmoid(scale)


class EfficientNetBiFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: EfficientNet-B0
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, features_only=True
        )

        # Feature Channels for EfficientNet-B0
        # idx 2: Stride 8 (P3) -> 40 channels
        # idx 3: Stride 16 (P4) -> 112 channels
        # idx 4: Stride 32 (P5) -> 320 channels

        # Projection Layers (1x1 Conv) to unify channel dimension
        self.proj_p3 = nn.Conv2d(40, Config.NECK_CHANNELS, 1, bias=False)
        self.proj_p4 = nn.Conv2d(112, Config.NECK_CHANNELS, 1, bias=False)
        self.proj_p5 = nn.Conv2d(320, Config.NECK_CHANNELS, 1, bias=False)

        self.bn_p3 = nn.BatchNorm2d(Config.NECK_CHANNELS)
        self.bn_p4 = nn.BatchNorm2d(Config.NECK_CHANNELS)
        self.bn_p5 = nn.BatchNorm2d(Config.NECK_CHANNELS)

        # Generate P6 and P7 from P5
        self.conv_p6 = nn.Conv2d(
            Config.NECK_CHANNELS,
            Config.NECK_CHANNELS,
            3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn_p6 = nn.BatchNorm2d(Config.NECK_CHANNELS)
        self.conv_p7 = nn.Conv2d(
            Config.NECK_CHANNELS,
            Config.NECK_CHANNELS,
            3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn_p7 = nn.BatchNorm2d(Config.NECK_CHANNELS)

        # BiFPN Neck
        self.bifpn = BiFPNBlock(Config.NECK_CHANNELS)

        # Upsampling Module (P3 Stride 8 -> Output Stride 4)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                Config.NECK_CHANNELS, Config.NECK_CHANNELS, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(Config.NECK_CHANNELS),
            Swish(),
        )

        # --- Task-Specific Split-Neck ---

        # 1. Classification Branch (Translation Invariant)
        self.cls_branch = nn.Sequential(
            nn.Conv2d(
                Config.NECK_CHANNELS, Config.NECK_CHANNELS, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(Config.NECK_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Regression Branch (Spatially Aware)
        self.reg_branch = nn.Sequential(
            CoordConv(Config.NECK_CHANNELS, Config.NECK_CHANNELS),
            nn.BatchNorm2d(Config.NECK_CHANNELS),
            nn.ReLU(inplace=True),
            SpatialAttention(),
        )

        # --- Heads ---
        self.head_heatmap = nn.Conv2d(Config.NECK_CHANNELS, Config.NUM_CLASSES, 1)
        self.head_wh = nn.Conv2d(Config.NECK_CHANNELS, 2, 1)
        self.head_offset = nn.Conv2d(Config.NECK_CHANNELS, 2, 1)

        # Global Head (Auxiliary) attached to P7
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(Config.NECK_CHANNELS, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize Heatmap Bias for Focal Loss
        self.head_heatmap.bias.data.fill_(-2.19)

        # Initialize Regression Heads with tiny std to prevent exploding gradients
        for head in [self.head_wh, self.head_offset]:
            nn.init.normal_(head.weight, std=Config.HEAD_INIT_STD)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        # Extract Backbone Features
        feats = self.backbone(x)
        # feats[2]=P3, feats[3]=P4, feats[4]=P5

        # Project to Neck Channels
        p3 = self.bn_p3(self.proj_p3(feats[2]))
        p4 = self.bn_p4(self.proj_p4(feats[3]))
        p5 = self.bn_p5(self.proj_p5(feats[4]))

        # Generate P6, P7
        p6 = self.bn_p6(self.conv_p6(p5))
        p7 = self.bn_p7(self.conv_p7(p6))

        # BiFPN Fusion
        p3_out, _, _, _, p7_out = self.bifpn(p3, p4, p5, p6, p7)

        # Upsample to Stride 4 (High Resolution)
        neck_out = self.upsample(p3_out)

        # Split Neck Processing
        feat_cls = self.cls_branch(neck_out)
        feat_reg = self.reg_branch(neck_out)

        # Predictions
        heatmap = self.head_heatmap(feat_cls)
        wh = self.head_wh(feat_reg)
        offset = self.head_offset(feat_reg)

        # Global Classification (Finding vs No Finding)
        global_logits = self.global_head(p7_out)

        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset,
            "global_logits": global_logits,
        }
