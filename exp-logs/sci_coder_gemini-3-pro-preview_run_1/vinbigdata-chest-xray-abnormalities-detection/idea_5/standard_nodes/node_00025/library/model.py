import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CoordConv(nn.Module):
    """
    Coordinate Convolution Layer.
    Concatenates normalized X and Y coordinate channels to the input features.
    This provides the model with explicit spatial awareness, which is critical
    for accurate regression of bounding box sizes and offsets.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CoordConv, self).__init__()
        # Input channels + 2 (one for X coords, one for Y coords)
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size, padding=padding, bias=True
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Create coordinate grids
        # Normalized to range [-1, 1]
        y_coords = (
            2.0 * torch.arange(h, device=x.device).unsqueeze(1).expand(h, w) / (h - 1.0)
            - 1.0
        )
        x_coords = (
            2.0 * torch.arange(w, device=x.device).unsqueeze(0).expand(h, w) / (w - 1.0)
            - 1.0
        )

        # Expand to batch size and reshape to (B, 1, H, W)
        y_coords = y_coords.view(1, 1, h, w).repeat(b, 1, 1, 1)
        x_coords = x_coords.view(1, 1, h, w).repeat(b, 1, 1, 1)

        # Concatenate original features with coordinate channels
        out = torch.cat([x, x_coords, y_coords], dim=1)

        return self.conv(out)


class SpatiallyAwareCenterNet(nn.Module):
    """
    CenterNet architecture with EfficientNet-B0 backbone, FPN Neck,
    and Spatially-Aware Decoupled Heads.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(SpatiallyAwareCenterNet, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # We extract features from strides 4, 8, 16, 32
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get channel counts for the extracted features
        # For EfficientNet-B0: [24, 40, 112, 320]
        feature_info = self.backbone.feature_info
        channels = [feature_info[i]["num_chs"] for i in (1, 2, 3, 4)]
        c2, c3, c4, c5 = channels

        # 2. Neck: FPN-style Upsampling
        # Project all levels to a common dimension (64) and fuse top-down
        self.neck_channels = 64

        self.lat5 = nn.Conv2d(c5, self.neck_channels, 1)
        self.lat4 = nn.Conv2d(c4, self.neck_channels, 1)
        self.lat3 = nn.Conv2d(c3, self.neck_channels, 1)
        self.lat2 = nn.Conv2d(c2, self.neck_channels, 1)

        # 3. Decoupled Heads

        # A. Heatmap Head (Classification)
        # Standard Conv -> ReLU -> Conv
        self.hm_head = nn.Sequential(
            nn.Conv2d(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, num_classes, 1),
        )

        # B. Size Head (Regression) - Spatially Aware
        # Uses CoordConv to inject spatial priors
        self.wh_head = nn.Sequential(
            CoordConv(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # C. Offset Head (Regression) - Spatially Aware
        # Uses CoordConv to inject spatial priors
        self.reg_head = nn.Sequential(
            CoordConv(self.neck_channels, self.neck_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.neck_channels, 2, 1),
        )

        # D. Global Classification Head (Finding vs No Finding)
        # Attached to the deepest feature map (C5)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize Heatmap Head bias for focal loss stability
        # -2.19 corresponds to a prior probability of 0.1
        self.hm_head[-1].bias.data.fill_(-2.19)

        # Initialize Regression Heads
        # Use a tiny standard deviation to prevent exploding gradients at start
        for head in [self.wh_head, self.reg_head]:
            last_layer = head[-1]
            last_layer.weight.data.normal_(0, 0.001)
            last_layer.bias.data.zero_()

    def forward(self, x):
        # Extract Backbone Features
        # c2: stride 4, c3: stride 8, c4: stride 16, c5: stride 32
        features = self.backbone(x)
        c2, c3, c4, c5 = features

        # Global Classification (on deepest feature C5)
        global_pred = self.global_head(c5)

        # Neck: Top-down pathway
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # Heads (on stride 4 feature map P2)
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
