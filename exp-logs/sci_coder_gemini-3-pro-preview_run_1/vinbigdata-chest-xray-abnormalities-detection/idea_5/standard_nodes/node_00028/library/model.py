import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class WeightedFusion(nn.Module):
    """
    Weighted Feature Fusion Block (BiFPN-inspired).
    Fuses current level features with upsampled features from the previous level
    using learnable scalar weights.
    """

    def __init__(self, channels):
        super(WeightedFusion, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_curr, x_prev_up):
        w1 = nn.functional.relu(self.w1)
        w2 = nn.functional.relu(self.w2)
        weights = w1 + w2 + self.epsilon

        # Fast normalized fusion
        out = (w1 * x_curr + w2 * x_prev_up) / weights
        return self.conv(out)


class SpatiallyAwareCenterNet(nn.Module):
    """
    CenterNet architecture with EfficientNet-B0 backbone and Weighted BiFPN-style Neck.
    Replaces explicit CoordConv with robust multi-scale fusion (Cite solution_lesson_node_00027).
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(SpatiallyAwareCenterNet, self).__init__()

        # 1. Backbone: EfficientNet-B0
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

        # 2. Neck: Weighted Feature Fusion (BiFPN-style Top-Down)
        self.neck_channels = 64

        # Lateral projections
        self.lat5 = nn.Conv2d(c5, self.neck_channels, 1)
        self.lat4 = nn.Conv2d(c4, self.neck_channels, 1)
        self.lat3 = nn.Conv2d(c3, self.neck_channels, 1)
        self.lat2 = nn.Conv2d(c2, self.neck_channels, 1)

        # Weighted Fusion Blocks
        self.fuse4 = WeightedFusion(self.neck_channels)
        self.fuse3 = WeightedFusion(self.neck_channels)
        self.fuse2 = WeightedFusion(self.neck_channels)

        # 3. Decoupled Heads (Standard Conv2d, no CoordConv)

        # A. Heatmap Head
        self.hm_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, num_classes, 1),
        )

        # B. Size Head
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # C. Offset Head
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # D. Global Classification Head
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

        # Initialize Regression Heads (Cite solution_lesson_node_00023)
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

        # Neck: Weighted Top-Down Fusion
        p5 = self.lat5(c5)

        p4_in = self.lat4(c4)
        p4 = self.fuse4(p4_in, F.interpolate(p5, scale_factor=2, mode="nearest"))

        p3_in = self.lat3(c3)
        p3 = self.fuse3(p3_in, F.interpolate(p4, scale_factor=2, mode="nearest"))

        p2_in = self.lat2(c2)
        p2 = self.fuse2(p2_in, F.interpolate(p3, scale_factor=2, mode="nearest"))

        # Heads
        hm = self.hm_head(p2)
        wh = self.wh_head(p2)
        reg = self.reg_head(p2)

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
