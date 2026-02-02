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
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
        Returns:
            Tensor of shape (N, C+2, H, W)
        """
        batch_size, _, h, w = x.shape

        # Create meshgrid
        # Normalized coordinates in range [-1, 1]
        y_range = torch.linspace(-1, 1, steps=h, device=x.device)
        x_range = torch.linspace(-1, 1, steps=w, device=x.device)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        # Expand to batch size
        y_grid = y_grid.expand(batch_size, 1, h, w)
        x_grid = x_grid.expand(batch_size, 1, h, w)

        # Concatenate
        out = torch.cat([x, x_grid, y_grid], dim=1)
        return out


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block with BN and Swish (SiLU).
    Used in BiFPN.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    Simplified BiFPN Block.
    Fuses features from P2, P3, P4, P5.
    """

    def __init__(self, feature_channels=64):
        super().__init__()
        self.epsilon = 1e-4

        # Learnable weights for fusion (3 inputs for some nodes, 2 for others)
        # We initialize them to 1.0 (will be normalized by Softmax-like logic or simple sum)
        # Here we use the "Fast normalized fusion" approach: w_i >= 0, sum = w_i / (sum(w_j) + eps)
        # Implemented as scalar parameters.

        # P4_td: Input P4_in, P5_in (resized)
        self.w_p4_td = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # P3_td: Input P3_in, P4_td (resized)
        self.w_p3_td = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # P2_td: Input P2_in, P3_td (resized)
        self.w_p2_td = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # P3_out: Input P3_in, P3_td, P2_out (resized) -> But P2 is bottom, so P2_out goes up to P3
        # Wait, BiFPN usually goes P3->P7. Here we have P2->P5.
        # Bottom-up path:
        # P3_out: Inputs P3_in, P3_td, P2_td (downsampled? No, P2 is high res, P3 is low res)
        # P2 is stride 4. P3 is stride 8.
        # To go P2 -> P3, we downsample.

        self.w_p3_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_p4_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_p5_out = nn.Parameter(torch.ones(3, dtype=torch.float32))

        # Convolutions after fusion
        self.conv_p4_td = SeparableConvBlock(feature_channels, feature_channels)
        self.conv_p3_td = SeparableConvBlock(feature_channels, feature_channels)
        self.conv_p2_td = SeparableConvBlock(feature_channels, feature_channels)

        self.conv_p3_out = SeparableConvBlock(feature_channels, feature_channels)
        self.conv_p4_out = SeparableConvBlock(feature_channels, feature_channels)
        self.conv_p5_out = SeparableConvBlock(feature_channels, feature_channels)

        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        # inputs: [p2, p3, p4, p5]
        p2_in, p3_in, p4_in, p5_in = inputs

        # --- Top-Down Path ---

        # P4_td = Conv(w1*P4_in + w2*Resize(P5_in))
        w_p4 = torch.relu(self.w_p4_td)
        w_p4 = w_p4 / (torch.sum(w_p4) + self.epsilon)
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest")
        p4_td = self.conv_p4_td(w_p4[0] * p4_in + w_p4[1] * p5_up)

        # P3_td = Conv(w1*P3_in + w2*Resize(P4_td))
        w_p3 = torch.relu(self.w_p3_td)
        w_p3 = w_p3 / (torch.sum(w_p3) + self.epsilon)
        p4_up = F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest")
        p3_td = self.conv_p3_td(w_p3[0] * p3_in + w_p3[1] * p4_up)

        # P2_td = Conv(w1*P2_in + w2*Resize(P3_td))
        w_p2 = torch.relu(self.w_p2_td)
        w_p2 = w_p2 / (torch.sum(w_p2) + self.epsilon)
        p3_up = F.interpolate(p3_td, size=p2_in.shape[-2:], mode="nearest")
        p2_td = self.conv_p2_td(w_p2[0] * p2_in + w_p2[1] * p3_up)

        # --- Bottom-Up Path ---
        # For CenterNet, we primarily want the high-res output (P2).
        # However, to complete the BiFPN structure, we compute up to P5.

        # P3_out = Conv(w1*P3_in + w2*P3_td + w3*Downsample(P2_td))
        w_p3_o = torch.relu(self.w_p3_out)
        w_p3_o = w_p3_o / (torch.sum(w_p3_o) + self.epsilon)
        p2_down = self.downsample(p2_td)
        # Ensure sizes match (handling odd dimensions)
        if p2_down.shape[-2:] != p3_in.shape[-2:]:
            p2_down = F.interpolate(p2_down, size=p3_in.shape[-2:], mode="nearest")
        p3_out = self.conv_p3_out(
            w_p3_o[0] * p3_in + w_p3_o[1] * p3_td + w_p3_o[2] * p2_down
        )

        # P4_out
        w_p4_o = torch.relu(self.w_p4_out)
        w_p4_o = w_p4_o / (torch.sum(w_p4_o) + self.epsilon)
        p3_down = self.downsample(p3_out)
        if p3_down.shape[-2:] != p4_in.shape[-2:]:
            p3_down = F.interpolate(p3_down, size=p4_in.shape[-2:], mode="nearest")
        p4_out = self.conv_p4_out(
            w_p4_o[0] * p4_in + w_p4_o[1] * p4_td + w_p4_o[2] * p3_down
        )

        # P5_out
        w_p5_o = torch.relu(self.w_p5_out)
        w_p5_o = w_p5_o / (torch.sum(w_p5_o) + self.epsilon)
        p4_down = self.downsample(p4_out)
        if p4_down.shape[-2:] != p5_in.shape[-2:]:
            p4_down = F.interpolate(p4_down, size=p5_in.shape[-2:], mode="nearest")
        p5_out = self.conv_p5_out(
            w_p5_o[0] * p5_in + w_p5_o[1] * p5_up + w_p5_o[2] * p4_down
        )  # Note: p5_up is just p5_in, using p5_in twice effectively

        return [p2_td, p3_out, p4_out, p5_out]


class ThoracicModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone (EfficientNet-B0)
        # features_only=True returns feature maps at different strides
        # Indices: 1 (P2, s4), 2 (P3, s8), 3 (P4, s16), 4 (P5, s32)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get channel counts
        dummy = torch.randn(1, 3, 256, 256)
        feats = self.backbone(dummy)
        c2, c3, c4, c5 = [f.shape[1] for f in feats]

        # 2. Neck (BiFPN Projection & Block)
        self.bifpn_channels = 64

        # Lateral convolutions to project backbone features to BiFPN size
        self.lat_p2 = nn.Conv2d(c2, self.bifpn_channels, 1)
        self.lat_p3 = nn.Conv2d(c3, self.bifpn_channels, 1)
        self.lat_p4 = nn.Conv2d(c4, self.bifpn_channels, 1)
        self.lat_p5 = nn.Conv2d(c5, self.bifpn_channels, 1)

        self.bifpn = BiFPNBlock(self.bifpn_channels)

        # 3. Decoupled Coordinate-Aware Heads

        # A. Heatmap Branch (Standard Conv)
        # Input: BiFPN P2 (Stride 4)
        self.hm_branch = nn.Sequential(
            nn.Conv2d(
                self.bifpn_channels, self.bifpn_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.bifpn_channels),
            nn.ReLU(inplace=True),
        )
        # Heatmap Output: (Num Classes - 1) for findings
        self.hm_head = nn.Conv2d(self.bifpn_channels, Config.NUM_CLASSES - 1, 1)

        # B. Regression Branch (CoordConv)
        # Input: BiFPN P2 (Stride 4)
        self.coord_conv = CoordConv()
        # Input channels = bifpn_channels + 2 (x, y)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(
                self.bifpn_channels + 2, self.bifpn_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.bifpn_channels),
            nn.ReLU(inplace=True),
        )

        # Size Head (Width, Height)
        self.wh_head = nn.Conv2d(self.bifpn_channels, 2, 1)

        # Offset Head (x_off, y_off)
        self.off_head = nn.Conv2d(self.bifpn_channels, 2, 1)

        # 4. Global Classification Head
        # Input: BiFPN P5 (Stride 32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Linear(self.bifpn_channels, 1)

        # 5. Initialization
        self._init_weights()

    def _init_weights(self):
        # Heatmap Bias Init (Focal Loss prior)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.hm_head.bias, bias_value)

        # Regression Heads Init (Tiny Std, Zero Bias)
        # Size Head
        nn.init.normal_(self.wh_head.weight, std=Config.HEAD_INIT_STD)
        nn.init.constant_(self.wh_head.bias, 0)

        # Offset Head
        nn.init.normal_(self.off_head.weight, std=Config.HEAD_INIT_STD)
        nn.init.constant_(self.off_head.bias, 0)

        # Global Head
        nn.init.xavier_uniform_(self.global_head.weight)
        nn.init.constant_(self.global_head.bias, 0)

    def forward(self, x):
        # 1. Backbone
        # [P2, P3, P4, P5]
        features = self.backbone(x)
        p2, p3, p4, p5 = features

        # 2. Projection
        p2 = self.lat_p2(p2)
        p3 = self.lat_p3(p3)
        p4 = self.lat_p4(p4)
        p5 = self.lat_p5(p5)

        # 3. BiFPN
        # Returns [p2_out, p3_out, p4_out, p5_out]
        fpn_feats = self.bifpn([p2, p3, p4, p5])

        # For CenterNet heads, we use the highest resolution feature map (P2, stride 4)
        # This P2 output from BiFPN has fused information from deep layers.
        feature_map = fpn_feats[0]

        # Deepest feature for global classification
        deep_feat = fpn_feats[3]

        # 4. Heads

        # A. Heatmap
        hm_feat = self.hm_branch(feature_map)
        heatmap = torch.sigmoid(self.hm_head(hm_feat))

        # B. Regression (CoordConv)
        # Add coordinates
        reg_input = self.coord_conv(feature_map)
        reg_feat = self.reg_branch(reg_input)

        # Size (Unbounded, but usually positive. Model predicts raw, loss handles it)
        size = self.wh_head(reg_feat)

        # Offset (Unbounded raw output, usually small values)
        offset = self.off_head(reg_feat)

        # C. Global Classification
        global_feat = self.global_pool(deep_feat).flatten(1)
        global_prob = torch.sigmoid(self.global_head(global_feat))

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset,
            "global_prob": global_prob,
        }
