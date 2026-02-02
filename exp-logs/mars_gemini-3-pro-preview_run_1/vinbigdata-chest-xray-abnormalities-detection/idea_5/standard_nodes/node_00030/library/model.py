import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution used in BiFPN.
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

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
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation
        if activation:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    Weighted Bi-directional Feature Pyramid Network Block.
    Fuses features from P3, P4, P5.
    """

    def __init__(self, channels):
        super(BiFPNBlock, self).__init__()
        self.epsilon = 1e-4

        # Weights for fusion (learnable scalars)
        # We use ReLU to ensure they are non-negative
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # Convolutions after fusion
        self.conv4_td = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv3_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv4_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )
        self.conv5_out = SeparableConvBlock(
            channels, channels, norm=True, activation=True
        )

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, p3, p4, p5):
        # --- Top-Down Pathway ---
        # P4_td = Conv(w1[0]*P4 + w1[1]*Resize(P5))
        w1 = torch.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.epsilon)
        p4_td = self.conv4_td(w1[0] * p4 + w1[1] * self.up(p5))

        # P3_out = Conv(w2[0]*P3 + w2[1]*Resize(P4_td))
        w2 = torch.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.epsilon)
        p3_out = self.conv3_out(w2[0] * p3 + w2[1] * self.up(p4_td))

        # --- Bottom-Up Pathway ---
        # P4_out = Conv(w3[0]*P4 + w3[1]*P4_td + w3[2]*Down(P3_out))
        w3 = torch.relu(self.w3)
        w3 = w3 / (torch.sum(w3) + self.epsilon)
        p4_out = self.conv4_out(w3[0] * p4 + w3[1] * p4_td + w3[2] * self.down(p3_out))

        # P5_out = Conv(w4[0]*P5 + w4[1]*Down(P4_out))
        w4 = torch.relu(self.w4)
        w4 = w4 / (torch.sum(w4) + self.epsilon)
        p5_out = self.conv5_out(w4[0] * p5 + w4[1] * self.down(p4_out))

        return p3_out, p4_out, p5_out


class BiFPNCenterNet(nn.Module):
    """
    CenterNet architecture with EfficientNet-B0 backbone and BiFPN Neck.
    Replaces the previous SpatiallyAwareCenterNet (FPN+CoordConv).
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(BiFPNCenterNet, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # We extract features from strides 4, 8, 16, 32
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get channel counts
        feature_info = self.backbone.feature_info
        channels = [feature_info[i]["num_chs"] for i in (1, 2, 3, 4)]
        c2, c3, c4, c5 = channels

        # 2. Neck: BiFPN
        self.neck_channels = 64

        # Projections to neck channels
        self.proj_c2 = nn.Conv2d(c2, self.neck_channels, 1)
        self.proj_c3 = nn.Conv2d(c3, self.neck_channels, 1)
        self.proj_c4 = nn.Conv2d(c4, self.neck_channels, 1)
        self.proj_c5 = nn.Conv2d(c5, self.neck_channels, 1)

        # BiFPN Block (Single Layer for efficiency)
        self.bifpn = BiFPNBlock(self.neck_channels)

        # 3. Heads
        # We fuse BiFPN output (P3) with C2 to get P2 (stride 4) for heads
        self.conv_p2 = SeparableConvBlock(
            self.neck_channels, self.neck_channels, norm=True, activation=True
        )

        # Standard CenterNet Heads (No CoordConv)
        # Heatmap
        self.hm_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, num_classes, 1),
        )

        # Size
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # Offset
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # Global Classification (on deepest feature C5)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize Heatmap Head bias
        self.hm_head[-1].bias.data.fill_(-2.19)

        # Initialize Regression Heads
        for head in [self.wh_head, self.reg_head]:
            last_layer = head[-1]
            last_layer.weight.data.normal_(0, 0.001)
            last_layer.bias.data.zero_()

    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        c2, c3, c4, c5 = features

        # Global Classification
        global_pred = self.global_head(c5)

        # Projections
        p2_in = self.proj_c2(c2)
        p3_in = self.proj_c3(c3)
        p4_in = self.proj_c4(c4)
        p5_in = self.proj_c5(c5)

        # BiFPN
        p3_out, p4_out, p5_out = self.bifpn(p3_in, p4_in, p5_in)

        # Final Fusion for Heads (Stride 4)
        # Upsample P3_out and add to P2_in
        p2_out = self.conv_p2(
            p2_in + F.interpolate(p3_out, scale_factor=2, mode="nearest")
        )

        # Heads
        hm = self.hm_head(p2_out)
        wh = self.wh_head(p2_out)
        reg = self.reg_head(p2_out)

        return {"hm": hm, "wh": wh, "reg": reg, "global": global_pred}


class CenterNetLoss(nn.Module):
    """
    Multi-task loss for CenterNet:
    1. Modified Focal Loss for Heatmap
    2. L1 Loss for Size (masked)
    3. L1 Loss for Offset (masked)
    4. BCE Loss for Global Classification
    """

    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.lambda_hm = Config.LAMBDA_HEATMAP
        self.lambda_wh = Config.LAMBDA_SIZE
        self.lambda_reg = Config.LAMBDA_OFFSET
        self.lambda_global = Config.LAMBDA_GLOBAL

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, batch):
        pred_hm = outputs["hm"]
        pred_wh = outputs["wh"]
        pred_reg = outputs["reg"]
        pred_global = outputs["global"]

        target_hm = batch["target_heatmap"].to(pred_hm.device)
        target_wh = batch["target_size"].to(pred_wh.device)
        target_reg = batch["target_offset"].to(pred_reg.device)
        target_mask = batch["target_mask"].to(pred_hm.device)
        target_global = batch["global_label"].to(pred_global.device)

        # 1. Heatmap Loss (Modified Focal Loss)
        # Apply sigmoid to logits before calculating loss
        pred_hm = torch.sigmoid(pred_hm)
        hm_loss = self._modified_focal_loss(pred_hm, target_hm)

        # 2. Regression Losses (L1)
        # Mask out predictions where there is no object
        mask = target_mask.unsqueeze(1).expand_as(pred_wh)

        # Sum loss and normalize by number of objects
        num_objs = mask.sum() + 1e-4

        wh_loss = (
            F.l1_loss(pred_wh * mask, target_wh * mask, reduction="sum") / num_objs
        )
        reg_loss = (
            F.l1_loss(pred_reg * mask, target_reg * mask, reduction="sum") / num_objs
        )

        # 3. Global Classification Loss
        global_loss = self.bce(pred_global, target_global)

        # Total Weighted Loss
        loss = (
            self.lambda_hm * hm_loss
            + self.lambda_wh * wh_loss
            + self.lambda_reg * reg_loss
            + self.lambda_global * global_loss
        )

        return loss, {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "wh_loss": wh_loss.item(),
            "reg_loss": reg_loss.item(),
            "global_loss": global_loss.item(),
        }

    def _modified_focal_loss(self, pred, gt):
        """
        Modified focal loss from CornerNet/CenterNet papers.
        Penalizes easy negatives less and focuses on hard examples.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples (penalty reduced near the center)
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Positive loss
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 2) * pos_inds

        # Negative loss
        neg_loss = (
            torch.log(1 - pred + 1e-12) * torch.pow(pred, 2) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss
