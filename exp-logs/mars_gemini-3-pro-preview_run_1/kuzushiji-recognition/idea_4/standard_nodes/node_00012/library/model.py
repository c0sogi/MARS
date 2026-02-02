import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import math
from library.config import Config


class DeformableConv2d(nn.Module):
    """
    A wrapper for Deformable Convolution v2 that handles the offset generation internally.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = 1

        # Offset convolution: predicts 2*k*k offsets
        # We are not using modulation (mask) here for simplicity, or we can add it.
        # Standard DCNv2 usually has modulation. torchvision DeformConv2d supports it if mask is passed.
        # Let's stick to standard offsets for robustness with available ops.
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding,
            stride=self.stride,
            bias=True,
        )

        self.dcn = torchvision.ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=self.stride,
            bias=bias,
        )

        # Initialize offsets to 0
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)


class DeformableHead(nn.Module):
    """
    Prediction head using Deformable Convolution.
    Structure: DCN -> BN -> ReLU -> Conv1x1
    """

    def __init__(self, in_channels, out_channels, head_conv=256):
        super(DeformableHead, self).__init__()
        self.dcn = DeformableConv2d(in_channels, head_conv, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(head_conv)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(
            head_conv, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.dcn(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenterNetConvNeXt(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(CenterNetConvNeXt, self).__init__()
        self.num_classes = num_classes

        # 1. Backbone: ConvNeXt Base
        # Features: [128, 256, 512, 1024] at strides [4, 8, 16, 32]
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, features_only=True
        )

        # Get channel counts from backbone
        dummy_input = torch.randn(1, 3, 256, 256)
        features = self.backbone(dummy_input)
        channels = [f.shape[1] for f in features]  # [128, 256, 512, 1024]

        # 2. Neck: FPN
        self.fpn_channels = 256

        # Lateral layers (project to 256)
        self.lateral0 = nn.Conv2d(channels[0], self.fpn_channels, 1)
        self.lateral1 = nn.Conv2d(channels[1], self.fpn_channels, 1)
        self.lateral2 = nn.Conv2d(channels[2], self.fpn_channels, 1)
        self.lateral3 = nn.Conv2d(channels[3], self.fpn_channels, 1)

        # 3. Heads
        # hm: Heatmap (1 channel for objectness)
        # cls: Classification (num_classes channels)
        # wh: Width/Height (2 channels)
        # reg: Regression/Offset (2 channels)

        self.hm_head = DeformableHead(self.fpn_channels, 1)
        self.cls_head = DeformableHead(self.fpn_channels, self.num_classes)
        self.wh_head = DeformableHead(self.fpn_channels, 2)
        self.reg_head = DeformableHead(self.fpn_channels, 2)

        # Init weights for heads
        self.init_weights()

    def init_weights(self):
        # Initialize heatmap head bias for focal loss stability
        # -2.19 corresponds to sigmoid(-2.19) approx 0.1
        self.hm_head.out_conv.bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone
        # c0: s4, c1: s8, c2: s16, c3: s32
        features = self.backbone(x)
        c0, c1, c2, c3 = features[0], features[1], features[2], features[3]

        # FPN Top-Down
        p3 = self.lateral3(c3)

        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        p0 = self.lateral0(c0) + F.interpolate(p1, scale_factor=2, mode="nearest")

        # Output is p0 (stride 4)

        # Heads
        hm = self.hm_head(p0)
        cls_logits = self.cls_head(p0)
        wh = self.wh_head(p0)
        reg = self.reg_head(p0)

        return {"hm": hm, "cls": cls_logits, "wh": wh, "reg": reg}


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()

    def modified_focal_loss(self, pred, gt):
        """
        pred: (batch, 1, h, w) - raw logits
        gt: (batch, 1, h, w) - gaussian heatmap (0-1)
        """
        pred = torch.sigmoid(pred)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    def reg_l1_loss(self, pred, target, mask):
        """
        pred: (batch, max_objs, 2)
        target: (batch, max_objs, 2)
        mask: (batch, max_objs)
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def cls_loss(self, pred, target, mask):
        """
        pred: (batch, max_objs, num_classes)
        target: (batch, max_objs) - class indices
        mask: (batch, max_objs)
        """
        # Flatten batch and objects
        pred = pred.view(-1, pred.size(2))  # (B*N, C)
        target = target.view(-1)  # (B*N)
        mask = mask.view(-1).float()  # (B*N)

        loss = F.cross_entropy(pred, target, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-4)
        return loss

    def _gather_feat(self, feat, ind):
        """
        Gather features at specific indices.
        feat: (B, C, H, W)
        ind: (B, K)
        """
        dim = feat.size(1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        feat = feat.view(feat.size(0), dim, -1).permute(0, 2, 1)

        # Gather
        feat = feat.gather(1, ind)
        return feat

    def forward(self, outputs, batch):
        hm_pred = outputs["hm"]
        cls_pred = outputs["cls"]
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]

        hm_gt = batch["hm"].to(hm_pred.device)
        wh_gt = batch["wh"].to(wh_pred.device)
        reg_gt = batch["reg"].to(reg_pred.device)
        ind = batch["ind"].to(reg_pred.device)
        cat_gt = batch["cat"].to(reg_pred.device)
        reg_mask = batch["reg_mask"].to(reg_pred.device)

        # 1. Heatmap Loss
        loss_hm = self.modified_focal_loss(hm_pred, hm_gt)

        # 2. Gather predictions at ground truth centers
        wh_pred_gathered = self._gather_feat(wh_pred, ind)
        reg_pred_gathered = self._gather_feat(reg_pred, ind)
        cls_pred_gathered = self._gather_feat(cls_pred, ind)

        # 3. Regression Losses
        loss_wh = self.reg_l1_loss(wh_pred_gathered, wh_gt, reg_mask)
        loss_reg = self.reg_l1_loss(reg_pred_gathered, reg_gt, reg_mask)

        # 4. Classification Loss
        loss_cls = self.cls_loss(cls_pred_gathered, cat_gt, reg_mask)

        # Weighted sum
        # Weights can be tuned. Standard CenterNet: hm=1, wh=0.1, reg=1
        loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg + 1.0 * loss_cls

        loss_stats = {
            "loss": loss,
            "hm_loss": loss_hm,
            "wh_loss": loss_wh,
            "reg_loss": loss_reg,
            "cls_loss": loss_cls,
        }

        return loss, loss_stats
