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
    Depthwise Separable Convolution with BatchNorm and Swish.
    Used as the building block for the BiFPN.
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
        self.act = Swish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CoordConv(nn.Module):
    """
    Coordinate Convolution Layer.
    Concatenates normalized (x, y) coordinates to the feature map to provide explicit spatial priors.
    Used in the Regression branch to enable Translation Covariance.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Input channels + 2 (one for x-coords, one for y-coords)
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # Generate normalized coordinate grids in range [-1, 1]
        y_coords = (
            torch.linspace(-1, 1, h, device=x.device)
            .view(1, 1, h, 1)
            .expand(b, 1, h, w)
        )
        x_coords = (
            torch.linspace(-1, 1, w, device=x.device)
            .view(1, 1, 1, w)
            .expand(b, 1, h, w)
        )

        # Concatenate coordinates to the input features
        out = torch.cat([x, x_coords, y_coords], dim=1)

        return self.act(self.bn(self.conv(out)))


class BiFPNBlock(nn.Module):
    """
    A single layer of Bi-directional Feature Pyramid Network (BiFPN).
    Fuses features from levels P3 to P7 using Fast Normalized Fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.epsilon = 1e-4

        # Learnable fusion weights (initialized to 1)
        # Top-down path weights
        self.w_p6_td = nn.Parameter(torch.ones(2))
        self.w_p5_td = nn.Parameter(torch.ones(2))
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_td = nn.Parameter(torch.ones(2))

        # Bottom-up path weights
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(3))
        self.w_p6_out = nn.Parameter(torch.ones(3))
        self.w_p7_out = nn.Parameter(torch.ones(2))

        # Convolutions applied after fusion
        self.conv_p6_td = SeparableConvBlock(channels, channels)
        self.conv_p5_td = SeparableConvBlock(channels, channels)
        self.conv_p4_td = SeparableConvBlock(channels, channels)
        self.conv_p3_out = SeparableConvBlock(channels, channels)

        self.conv_p4_out = SeparableConvBlock(channels, channels)
        self.conv_p5_out = SeparableConvBlock(channels, channels)
        self.conv_p6_out = SeparableConvBlock(channels, channels)
        self.conv_p7_out = SeparableConvBlock(channels, channels)

    def forward(self, inputs):
        # inputs: [P3, P4, P5, P6, P7]
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # --- Top-Down Pathway ---

        # P7 -> P6
        w = torch.relu(self.w_p6_td)
        w = w / (torch.sum(w) + self.epsilon)
        p6_td = self.conv_p6_td(
            w[0] * p6_in + w[1] * F.interpolate(p7_in, scale_factor=2, mode="nearest")
        )

        # P6 -> P5
        w = torch.relu(self.w_p5_td)
        w = w / (torch.sum(w) + self.epsilon)
        p5_td = self.conv_p5_td(
            w[0] * p5_in + w[1] * F.interpolate(p6_td, scale_factor=2, mode="nearest")
        )

        # P5 -> P4
        w = torch.relu(self.w_p4_td)
        w = w / (torch.sum(w) + self.epsilon)
        p4_td = self.conv_p4_td(
            w[0] * p4_in + w[1] * F.interpolate(p5_td, scale_factor=2, mode="nearest")
        )

        # P4 -> P3 (This generates the initial P3 output)
        w = torch.relu(self.w_p3_td)
        w = w / (torch.sum(w) + self.epsilon)
        p3_out = self.conv_p3_out(
            w[0] * p3_in + w[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest")
        )

        # --- Bottom-Up Pathway ---

        # P3 -> P4
        w = torch.relu(self.w_p4_out)
        w = w / (torch.sum(w) + self.epsilon)
        p4_out = self.conv_p4_out(
            w[0] * p4_in
            + w[1] * p4_td
            + w[2] * F.interpolate(p3_out, scale_factor=0.5, mode="nearest")
        )

        # P4 -> P5
        w = torch.relu(self.w_p5_out)
        w = w / (torch.sum(w) + self.epsilon)
        p5_out = self.conv_p5_out(
            w[0] * p5_in
            + w[1] * p5_td
            + w[2] * F.interpolate(p4_out, scale_factor=0.5, mode="nearest")
        )

        # P5 -> P6
        w = torch.relu(self.w_p6_out)
        w = w / (torch.sum(w) + self.epsilon)
        p6_out = self.conv_p6_out(
            w[0] * p6_in
            + w[1] * p6_td
            + w[2] * F.interpolate(p5_out, scale_factor=0.5, mode="nearest")
        )

        # P6 -> P7
        w = torch.relu(self.w_p7_out)
        w = w / (torch.sum(w) + self.epsilon)
        p7_out = self.conv_p7_out(
            w[0] * p7_in
            + w[1] * F.interpolate(p6_out, scale_factor=0.5, mode="nearest")
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class DetModel(nn.Module):
    """
    Task-Aligned Spatially-Decoupled CenterNet.

    Architecture:
    1. Backbone: EfficientNet-B0 (P3, P4, P5)
    2. Neck: BiFPN (P3-P7) -> Upsampled to Stride 4
    3. Heads:
       - Classification: Standard Conv (Invariant)
       - Regression: CoordConv (Covariant)
       - Global: Aux Head on P7
    """

    def __init__(self, config=Config):
        super().__init__()
        self.num_classes = config.NUM_CLASSES
        self.backbone_name = config.BACKBONE
        self.pretrained = config.PRETRAINED

        # 1. Backbone
        # Extract P3 (stride 8), P4 (stride 16), P5 (stride 32)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Dynamically determine channel counts
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy)
        c3, c4, c5 = [f.shape[1] for f in feats]

        # 2. Neck: BiFPN
        self.bifpn_channels = 64
        self.num_bifpn_layers = 3  # Stack 3 BiFPN layers for robust fusion

        # Projections to align backbone channels to BiFPN width
        self.p3_proj = nn.Conv2d(c3, self.bifpn_channels, 1)
        self.p4_proj = nn.Conv2d(c4, self.bifpn_channels, 1)
        self.p5_proj = nn.Conv2d(c5, self.bifpn_channels, 1)

        # Generate P6 and P7 from P5
        self.p6_conv = nn.Conv2d(c5, self.bifpn_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(
            self.bifpn_channels, self.bifpn_channels, 3, stride=2, padding=1
        )

        # BiFPN Stack
        self.bifpn_layers = nn.ModuleList(
            [BiFPNBlock(self.bifpn_channels) for _ in range(self.num_bifpn_layers)]
        )

        # Final Upsampling Block: P3 (Stride 8) -> Stride 4
        self.upsample_to_stride4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.bifpn_channels, self.bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(self.bifpn_channels),
            nn.ReLU(inplace=True),
        )

        # 3. Heads

        # A. Classification Branch (Translation Invariant)
        # Standard convolutions preserve invariance.
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.bifpn_channels, self.bifpn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bifpn_channels, self.num_classes, 1),
        )

        # B. Regression Branch (Translation Covariant)
        # Uses CoordConv to inject spatial information.
        # Outputs 4 channels: 2 for Width/Height, 2 for Center Offset
        self.reg_head_adapter = CoordConv(self.bifpn_channels, self.bifpn_channels)
        self.reg_head_out = nn.Conv2d(self.bifpn_channels, 4, 1)

        # C. Auxiliary Global Head (Finding vs No Finding)
        # Attached to the deepest semantic layer (P7)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.bifpn_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize Classification Head
        # Bias initialization for Focal Loss stability: -log((1-pi)/pi)
        for m in self.cls_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Explicit bias init for the final classification layer
        nn.init.constant_(self.cls_head[-1].bias, -2.19)

        # Initialize Regression Head
        # Use tiny std to prevent exploding gradients in the early phase
        nn.init.normal_(self.reg_head_out.weight, std=0.001)
        nn.init.constant_(self.reg_head_out.bias, 0)

    def forward(self, x):
        # x: (B, 3, H, W)

        # 1. Backbone Feature Extraction
        p3, p4, p5 = self.backbone(x)

        # 2. Projection & Generation
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5_orig = p5
        p5 = self.p5_proj(p5)

        p6 = self.p6_conv(p5_orig)
        p7 = self.p7_conv(F.relu(p6))

        features = [p3, p4, p5, p6, p7]

        # 3. BiFPN Fusion
        for layer in self.bifpn_layers:
            features = layer(features)

        p3_out, p4_out, p5_out, p6_out, p7_out = features

        # 4. Upsample to Stride 4 (High Resolution for Dense Heads)
        neck_out = self.upsample_to_stride4(p3_out)

        # 5. Heads

        # Heatmap Prediction
        hm = self.cls_head(neck_out)

        # Regression Prediction (Split into WH and Offset)
        reg_feat = self.reg_head_adapter(neck_out)
        reg_out = self.reg_head_out(reg_feat)
        wh = reg_out[:, :2, :, :]
        offset = reg_out[:, 2:, :, :]

        # Global Classification (on P7)
        global_cls = self.global_head(p7_out)

        return {"hm": hm, "wh": wh, "reg": offset, "global_cls": global_cls}
