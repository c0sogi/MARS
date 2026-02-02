import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import NUM_CLASSES

# =============================================================================
# LAYERS & BLOCKS
# =============================================================================


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution with BatchNorm and Swish activation.
    Used inside BiFPN.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Swish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNLayer(nn.Module):
    """
    A single layer of Bi-directional Feature Pyramid Network.
    Fuses features P3, P4, P5, P6, P7.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.epsilon = 1e-4

        # Weights for fusion (learnable)
        # 5 levels: P3, P4, P5, P6, P7
        # Top-down path: P7->P6, P6->P5, P5->P4, P4->P3
        # Bottom-up path: P3->P4, P4->P5, P5->P6, P6->P7

        # Convolutions for fusion
        self.conv_p6_td = SeparableConvBlock(channels, channels)
        self.conv_p5_td = SeparableConvBlock(channels, channels)
        self.conv_p4_td = SeparableConvBlock(channels, channels)
        self.conv_p3_td = SeparableConvBlock(channels, channels)

        self.conv_p4_out = SeparableConvBlock(channels, channels)
        self.conv_p5_out = SeparableConvBlock(channels, channels)
        self.conv_p6_out = SeparableConvBlock(channels, channels)
        self.conv_p7_out = SeparableConvBlock(channels, channels)

        # Weight parameters
        self.w_p6_td = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_p5_td = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_p4_td = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_p3_td = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.w_p4_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_p5_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_p6_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_p7_out = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, inputs):
        """
        inputs: list of [P3, P4, P5, P6, P7]
        """
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # --- Top-Down Pathway ---

        # P7 -> P6
        w_p6 = torch.relu(self.w_p6_td)
        w_p6 = w_p6 / (torch.sum(w_p6) + self.epsilon)
        p6_td = self.conv_p6_td(
            w_p6[0] * p6_in
            + w_p6[1] * F.interpolate(p7_in, scale_factor=2, mode="nearest")
        )

        # P6 -> P5
        w_p5 = torch.relu(self.w_p5_td)
        w_p5 = w_p5 / (torch.sum(w_p5) + self.epsilon)
        p5_td = self.conv_p5_td(
            w_p5[0] * p5_in
            + w_p5[1] * F.interpolate(p6_td, scale_factor=2, mode="nearest")
        )

        # P5 -> P4
        w_p4 = torch.relu(self.w_p4_td)
        w_p4 = w_p4 / (torch.sum(w_p4) + self.epsilon)
        p4_td = self.conv_p4_td(
            w_p4[0] * p4_in
            + w_p4[1] * F.interpolate(p5_td, scale_factor=2, mode="nearest")
        )

        # P4 -> P3
        w_p3 = torch.relu(self.w_p3_td)
        w_p3 = w_p3 / (torch.sum(w_p3) + self.epsilon)
        p3_out = self.conv_p3_td(
            w_p3[0] * p3_in
            + w_p3[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest")
        )

        # --- Bottom-Up Pathway ---

        # P3 -> P4
        w_p4_up = torch.relu(self.w_p4_out)
        w_p4_up = w_p4_up / (torch.sum(w_p4_up) + self.epsilon)
        p4_out = self.conv_p4_out(
            w_p4_up[0] * p4_in
            + w_p4_up[1] * p4_td
            + w_p4_up[2] * F.max_pool2d(p3_out, kernel_size=3, stride=2, padding=1)
        )

        # P4 -> P5
        w_p5_up = torch.relu(self.w_p5_out)
        w_p5_up = w_p5_up / (torch.sum(w_p5_up) + self.epsilon)
        p5_out = self.conv_p5_out(
            w_p5_up[0] * p5_in
            + w_p5_up[1] * p5_td
            + w_p5_up[2] * F.max_pool2d(p4_out, kernel_size=3, stride=2, padding=1)
        )

        # P5 -> P6
        w_p6_up = torch.relu(self.w_p6_out)
        w_p6_up = w_p6_up / (torch.sum(w_p6_up) + self.epsilon)
        p6_out = self.conv_p6_out(
            w_p6_up[0] * p6_in
            + w_p6_up[1] * p6_td
            + w_p6_up[2] * F.max_pool2d(p5_out, kernel_size=3, stride=2, padding=1)
        )

        # P6 -> P7
        w_p7_up = torch.relu(self.w_p7_out)
        w_p7_up = w_p7_up / (torch.sum(w_p7_up) + self.epsilon)
        p7_out = self.conv_p7_out(
            w_p7_up[0] * p7_in
            + w_p7_up[1] * F.max_pool2d(p6_out, kernel_size=3, stride=2, padding=1)
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class CoordConv(nn.Module):
    """
    Coordinate Convolution: Concatenates normalized X and Y channels to the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size, padding=padding, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Create meshgrid
        y_range = torch.linspace(-1, 1, h, device=x.device)
        x_range = torch.linspace(-1, 1, w, device=x.device)
        yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

        # Expand to batch size
        xx = xx.expand(b, 1, h, w)
        yy = yy.expand(b, 1, h, w)

        # Concatenate
        x = torch.cat([x, xx, yy], dim=1)

        return self.conv(x)


# =============================================================================
# HEADS
# =============================================================================


class ClassificationHead(nn.Module):
    """
    Standard Conv Head for Heatmap Prediction (Translation Invariant).
    """

    def __init__(self, in_channels, num_classes, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = Swish()
        self.conv2 = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True)

        # Initialize final layer for focal loss background bias
        # bias = -log((1-pi)/pi) where pi=0.01
        self.conv2.bias.data.fill_(-4.6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class RegressionHead(nn.Module):
    """
    Split Regression Head with CoordConv Adapter.
    Outputs: Size (W, H) and Offset (dX, dY).
    """

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        # Shared Adapter with CoordConv
        self.adapter = nn.Sequential(
            CoordConv(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            Swish(),
        )

        # Split Heads
        self.size_head = nn.Conv2d(
            hidden_channels, 2, kernel_size=3, padding=1, bias=True
        )
        self.offset_head = nn.Conv2d(
            hidden_channels, 2, kernel_size=3, padding=1, bias=True
        )

        self._init_weights()

    def _init_weights(self):
        # Tiny std dev for regression heads to prevent gradient explosion
        nn.init.normal_(self.size_head.weight, std=0.001)
        nn.init.constant_(self.size_head.bias, 0)

        nn.init.normal_(self.offset_head.weight, std=0.001)
        nn.init.constant_(self.offset_head.bias, 0)

    def forward(self, x):
        feat = self.adapter(x)
        size = self.size_head(feat)
        offset = self.offset_head(feat)
        return size, offset


class GlobalHead(nn.Module):
    """
    Binary Classification Head on P7 for 'Finding vs No Finding'.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================


class EfficientDetDecoupled(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, bifpn_channels=64, num_bifpn_layers=3):
        super().__init__()

        # 1. Backbone: EfficientNet-B0
        # Indices (2, 3, 4) -> Strides 8, 16, 32 -> Channels 40, 112, 320
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        feature_info = self.backbone.feature_info.channels()  # [40, 112, 320]

        # 2. Projections to BiFPN width
        self.p3_proj = nn.Conv2d(feature_info[0], bifpn_channels, 1)
        self.p4_proj = nn.Conv2d(feature_info[1], bifpn_channels, 1)
        self.p5_proj = nn.Conv2d(feature_info[2], bifpn_channels, 1)

        # P6 and P7 generation
        # P6 is downsampled from P5
        self.p6_conv = nn.Conv2d(
            feature_info[2], bifpn_channels, 3, stride=2, padding=1
        )
        # P7 is downsampled from P6 (after p6_conv)
        self.p7_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, stride=2, padding=1)

        # 3. BiFPN Layers
        self.bifpn = nn.Sequential(
            *[BiFPNLayer(channels=bifpn_channels) for _ in range(num_bifpn_layers)]
        )

        # 4. Upsampling to Stride 4
        # P3 is Stride 8. We upsample by 2x.
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(bifpn_channels),
            Swish(),
        )

        # 5. Heads
        self.cls_head = ClassificationHead(bifpn_channels, num_classes)
        # Cite Lesson 36: Separate heads for disparate scales/tasks
        self.size_head = RegressionHead(bifpn_channels, num_dims=2)
        self.offset_head = RegressionHead(bifpn_channels, num_dims=2)
        self.global_head = GlobalHead(bifpn_channels)

    def forward(self, x):
        # Backbone
        feats = self.backbone(x)  # [P3, P4, P5]
        p3, p4, p5 = feats

        # Projection & Generation
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5_orig = p5  # Keep original for P6 generation
        p5 = self.p5_proj(p5)

        p6 = self.p6_conv(p5_orig)
        p7 = self.p7_conv(p6)

        # BiFPN
        features = [p3, p4, p5, p6, p7]
        for layer in self.bifpn:
            features = layer(features)

        p3_out, _, _, _, p7_out = features

        # Upsample P3 to Stride 4
        neck_out = self.upsample_conv(p3_out)

        # Predictions
        heatmap = self.cls_head(neck_out)
        size = self.size_head(neck_out)
        offset = self.offset_head(neck_out)
        global_logits = self.global_head(p7_out)

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset,
            "global_logits": global_logits,
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================


class ThoracicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def focal_loss(self, pred, gt):
        """
        Modified focal loss (CenterNet variant).
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(torch.sigmoid(pred), 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def reg_l1_loss(self, pred, target, mask):
        """
        L1 Loss masked by object presence.
        """
        loss = F.l1_loss(pred, target, reduction="none")
        loss = loss * mask
        num_pos = mask.sum() + 1e-4
        return loss.sum() / num_pos

    def forward(self, outputs, targets):
        # 1. Heatmap Loss
        hm_loss = self.focal_loss(outputs["heatmap"], targets["heatmap"])

        # 2. Regression Losses
        # Mask is (B, 1, H, W). Expand to (B, 2, H, W) for size/offset
        mask = targets["mask"].expand_as(targets["size"])

        size_loss = self.reg_l1_loss(outputs["size"], targets["size"], mask)
        offset_loss = self.reg_l1_loss(outputs["offset"], targets["offset"], mask)

        # 3. Global Classification Loss
        global_loss = self.bce(outputs["global_logits"], targets["global_label"])

        # Weighted Sum (Heuristics)
        # Heatmap is dominant. Size/Offset are auxiliary but critical. Global is gate.
        total_loss = hm_loss + 0.1 * size_loss + 1.0 * offset_loss + 0.5 * global_loss

        return total_loss, {
            "hm_loss": hm_loss.item(),
            "size_loss": size_loss.item(),
            "off_loss": offset_loss.item(),
            "glob_loss": global_loss.item(),
        }
