import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config
from library.utils import _transpose_and_gather_feat


class SwinCenterNet(nn.Module):
    """
    Swin Transformer (Base) backbone with a CenterNet Head and FPN Neck.
    """

    def __init__(self):
        super(SwinCenterNet, self).__init__()
        self.num_classes = Config.get_num_classes()

        # 1. Backbone: Swin-B
        # features_only=True returns features from the stages
        # out_indices=(0, 1, 2, 3) corresponds to strides 4, 8, 16, 32
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=Config.IMG_SIZE,
        )

        # 2. Neck: Feature Pyramid Network (FPN)
        # Project all encoder channels to FPN_OUT_CHANNELS (256)
        enc_channels = Config.ENCODER_CHANNELS  # [128, 256, 512, 1024]
        fpn_channels = Config.FPN_OUT_CHANNELS  # 256

        self.lat_layer1 = nn.Conv2d(enc_channels[0], fpn_channels, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(enc_channels[1], fpn_channels, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(enc_channels[2], fpn_channels, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(enc_channels[3], fpn_channels, kernel_size=1)

        # 3. Heads
        # Common pattern: 3x3 Conv -> ReLU -> 1x1 Conv

        # Heatmap Head (Objectness): 1 channel
        self.hm_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 1, kernel_size=1, bias=True),
        )

        # Width/Height Regression Head: 2 channels
        self.wh_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 2, kernel_size=1, bias=True),
        )

        # Local Offset Regression Head: 2 channels
        self.reg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 2, kernel_size=1, bias=True),
        )

        # Classification Head: NumClasses channels
        # We predict class logits at every pixel, but only train at centers
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, self.num_classes, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize FPN and Heads
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize Heatmap Bias to -2.19 (for Focal Loss stability)
        # -log((1 - pi) / pi) where pi = 0.1
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone Feature Extraction
        # c1: stride 4, c2: stride 8, c3: stride 16, c4: stride 32
        c1, c2, c3, c4 = self.backbone(x)

        # FPN Top-Down Pathway
        p4 = self.lat_layer4(c4)

        p3 = self.lat_layer3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        p2 = self.lat_layer2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # Final output is P1 (Stride 4)
        p1 = self.lat_layer1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        # Heads
        hm = self.hm_head(p1)
        wh = self.wh_head(p1)
        reg = self.reg_head(p1)
        cls_logits = self.cls_head(p1)

        return {"hm": hm, "wh": wh, "reg": reg, "cls_logits": cls_logits}


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = Config.HM_LOSS_WEIGHT
        self.wh_weight = Config.WH_LOSS_WEIGHT
        self.off_weight = Config.OFF_LOSS_WEIGHT
        self.cls_weight = Config.CLS_LOSS_WEIGHT

    def _neg_loss(self, pred, gt):
        """
        Modified Focal Loss for Heatmap.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp pred to avoid log(0)
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

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

    def _reg_loss(self, regr, gt_regr, mask):
        """
        L1 Loss for regression masked by object presence.
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()

        regr = regr * mask
        gt_regr = gt_regr * mask

        regr_loss = F.l1_loss(regr, gt_regr, reduction="sum")
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss

    def _cls_loss(self, cls_logits, gt_cls_ids, mask):
        """
        Cross Entropy Loss applied only at ground truth centers.
        """
        # cls_logits: (B, K, NumClasses)
        # gt_cls_ids: (B, K)
        # mask: (B, K)

        # Flatten for processing
        cls_logits = cls_logits.view(-1, cls_logits.size(-1))
        gt_cls_ids = gt_cls_ids.view(-1)
        mask = mask.view(-1).float()

        # Compute CE loss (no reduction to apply mask)
        loss = F.cross_entropy(cls_logits, gt_cls_ids, reduction="none")

        # Apply mask
        loss = (loss * mask).sum()

        # Normalize
        num_pos = mask.sum()
        loss = loss / (num_pos + 1e-4)

        return loss

    def forward(self, outputs, batch):
        # Predictions
        hm_pred = torch.sigmoid(outputs["hm"])
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]
        cls_logits_pred = outputs["cls_logits"]

        # Targets
        # Ensure targets are on the same device as predictions
        device = hm_pred.device
        hm_true = batch["hm"].to(device)
        wh_true = batch["wh"].to(device)
        reg_true = batch["reg"].to(device)
        ind_true = batch["ind"].to(device)
        cls_ids_true = batch["cls_ids"].to(device)
        reg_mask = batch["reg_mask"].to(device)

        # 1. Heatmap Loss
        hm_loss = self._neg_loss(hm_pred, hm_true)

        # 2. Gather features at ground truth centers
        # reg_pred: (B, 2, H, W) -> (B, K, 2)
        reg_pred_gathered = _transpose_and_gather_feat(reg_pred, ind_true)

        # wh_pred: (B, 2, H, W) -> (B, K, 2)
        wh_pred_gathered = _transpose_and_gather_feat(wh_pred, ind_true)

        # cls_logits_pred: (B, NumClasses, H, W) -> (B, K, NumClasses)
        cls_pred_gathered = _transpose_and_gather_feat(cls_logits_pred, ind_true)

        # 3. Regression Losses
        off_loss = self._reg_loss(reg_pred_gathered, reg_true, reg_mask)
        wh_loss = self._reg_loss(wh_pred_gathered, wh_true, reg_mask)

        # 4. Classification Loss
        cls_loss = self._cls_loss(cls_pred_gathered, cls_ids_true, reg_mask)

        # Weighted Sum
        total_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
            + self.cls_weight * cls_loss
        )

        return total_loss, {
            "loss": total_loss.item(),
            "hm_loss": hm_loss.item(),
            "wh_loss": wh_loss.item(),
            "off_loss": off_loss.item(),
            "cls_loss": cls_loss.item(),
        }
