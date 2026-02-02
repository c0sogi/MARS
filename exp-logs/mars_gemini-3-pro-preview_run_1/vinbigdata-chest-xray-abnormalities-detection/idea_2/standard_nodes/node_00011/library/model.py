import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution with BatchNorm and Swish activation.
    Standard building block for EfficientDet/BiFPN.
    """

    def __init__(self, in_channels, out_channels):
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
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Swish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    A single BiFPN layer that fuses features from levels P3 to P7.
    Implements weighted bi-directional fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.epsilon = 1e-4

        # Learnable weights for fusion (initialized to 1)
        # Top-down path
        self.w_p6_td = nn.Parameter(torch.ones(2))
        self.w_p5_td = nn.Parameter(torch.ones(2))
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))  # Bottom of U-shape

        # Bottom-up path
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(3))
        self.w_p6_out = nn.Parameter(torch.ones(3))
        self.w_p7_out = nn.Parameter(torch.ones(2))

        # Convolutions after fusion
        self.conv_p6_td = SeparableConvBlock(channels, channels)
        self.conv_p5_td = SeparableConvBlock(channels, channels)
        self.conv_p4_td = SeparableConvBlock(channels, channels)

        self.conv_p3_out = SeparableConvBlock(channels, channels)
        self.conv_p4_out = SeparableConvBlock(channels, channels)
        self.conv_p5_out = SeparableConvBlock(channels, channels)
        self.conv_p6_out = SeparableConvBlock(channels, channels)
        self.conv_p7_out = SeparableConvBlock(channels, channels)

        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs):
        # inputs: [P3, P4, P5, P6, P7]
        p3, p4, p5, p6, p7 = inputs

        # --- Top-Down Path ---

        # P6_td = Conv(w1*P6 + w2*Resize(P7))
        w = torch.relu(self.w_p6_td)
        w = w / (torch.sum(w) + self.epsilon)
        p6_td = self.conv_p6_td(w[0] * p6 + w[1] * self.upsample(p7))

        # P5_td = Conv(w1*P5 + w2*Resize(P6_td))
        w = torch.relu(self.w_p5_td)
        w = w / (torch.sum(w) + self.epsilon)
        p5_td = self.conv_p5_td(w[0] * p5 + w[1] * self.upsample(p6_td))

        # P4_td = Conv(w1*P4 + w2*Resize(P5_td))
        w = torch.relu(self.w_p4_td)
        w = w / (torch.sum(w) + self.epsilon)
        p4_td = self.conv_p4_td(w[0] * p4 + w[1] * self.upsample(p5_td))

        # --- Bottom-Up Path ---

        # P3_out = Conv(w1*P3 + w2*Resize(P4_td))
        w = torch.relu(self.w_p3_out)
        w = w / (torch.sum(w) + self.epsilon)
        p3_out = self.conv_p3_out(w[0] * p3 + w[1] * self.upsample(p4_td))

        # P4_out = Conv(w1*P4 + w2*P4_td + w3*Resize(P3_out))
        w = torch.relu(self.w_p4_out)
        w = w / (torch.sum(w) + self.epsilon)
        p4_out = self.conv_p4_out(
            w[0] * p4 + w[1] * p4_td + w[2] * self.downsample(p3_out)
        )

        # P5_out = Conv(w1*P5 + w2*P5_td + w3*Resize(P4_out))
        w = torch.relu(self.w_p5_out)
        w = w / (torch.sum(w) + self.epsilon)
        p5_out = self.conv_p5_out(
            w[0] * p5 + w[1] * p5_td + w[2] * self.downsample(p4_out)
        )

        # P6_out = Conv(w1*P6 + w2*P6_td + w3*Resize(P5_out))
        w = torch.relu(self.w_p6_out)
        w = w / (torch.sum(w) + self.epsilon)
        p6_out = self.conv_p6_out(
            w[0] * p6 + w[1] * p6_td + w[2] * self.downsample(p5_out)
        )

        # P7_out = Conv(w1*P7 + w2*Resize(P6_out))
        w = torch.relu(self.w_p7_out)
        w = w / (torch.sum(w) + self.epsilon)
        p7_out = self.conv_p7_out(w[0] * p7 + w[1] * self.downsample(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):
    def __init__(self, backbone_channels, fpn_channels=64, num_layers=2):
        super().__init__()
        # backbone_channels: [C3, C4, C5]
        self.fpn_channels = fpn_channels

        # Projections to align backbone channels to FPN channels
        self.p3_in = nn.Conv2d(backbone_channels[0], fpn_channels, 1)
        self.p4_in = nn.Conv2d(backbone_channels[1], fpn_channels, 1)
        self.p5_in = nn.Conv2d(backbone_channels[2], fpn_channels, 1)

        # Generate P6 and P7 from C5
        self.p6_in = nn.Conv2d(
            backbone_channels[2], fpn_channels, 3, stride=2, padding=1
        )
        self.p7_in = nn.MaxPool2d(3, stride=2, padding=1)

        # Stack BiFPN blocks
        self.layers = nn.ModuleList(
            [BiFPNBlock(fpn_channels) for _ in range(num_layers)]
        )

    def forward(self, features):
        c3, c4, c5 = features

        p3 = self.p3_in(c3)
        p4 = self.p4_in(c4)
        p5 = self.p5_in(c5)
        p6 = self.p6_in(c5)
        p7 = self.p7_in(p6)

        features = [p3, p4, p5, p6, p7]

        for layer in self.layers:
            features = layer(features)

        return features


class MultiTaskCenterNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone: EfficientNet-B0
        # out_indices=(2, 3, 4) corresponds to strides 8, 16, 32 (P3, P4, P5)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Determine backbone output channels dynamically
        dummy_input = torch.randn(1, 3, 256, 256)
        feats = self.backbone(dummy_input)
        backbone_channels = [f.shape[1] for f in feats]

        # 2. Neck: BiFPN
        self.fpn_channels = 64
        self.bifpn = BiFPN(
            backbone_channels, fpn_channels=self.fpn_channels, num_layers=2
        )

        # 3. Upsampling for CenterNet Heads
        # BiFPN outputs P3 (stride 8). CenterNet typically uses stride 4.
        # We upsample P3 by 2x.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                self.fpn_channels, self.fpn_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.fpn_channels),
            Swish(),
        )

        self.head_conv = 64

        # 4. Detection Heads
        # Heatmap Head (Classes 0-13)
        self.hm_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, Config.NUM_CLASSES, 1),
        )
        # Width/Height Head
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, 1),
        )
        # Regression/Offset Head
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, 1),
        )

        # 5. Global Classification Head (attached to P7)
        # Predicts if there is ANY finding vs No finding.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fpn_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize heatmap bias for Focal Loss stability (prob ~ 0.1 at start)
        self.hm_head[-1].bias.data.fill_(-2.19)

        # Initialize other heads
        for head in [self.wh_head, self.reg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        feats = self.backbone(x)  # [C3, C4, C5]

        # BiFPN Fusion
        fpn_feats = self.bifpn(feats)  # [P3, P4, P5, P6, P7]

        p3 = fpn_feats[0]  # Stride 8
        p7 = fpn_feats[4]  # Stride 128

        # Upsample P3 to Stride 4 for dense detection
        det_feats = self.upsample(p3)

        # Detection Outputs
        hm = self.hm_head(det_feats)
        wh = self.wh_head(det_feats)
        reg = self.reg_head(det_feats)

        # Global Classification Output
        global_logits = self.global_head(self.global_pool(p7))

        return {"hm": hm, "wh": wh, "reg": reg, "global_logits": global_logits}
