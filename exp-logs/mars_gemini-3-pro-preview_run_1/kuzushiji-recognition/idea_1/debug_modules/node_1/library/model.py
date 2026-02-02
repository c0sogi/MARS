import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        # Use weights='DEFAULT' for modern torchvision versions compatible with Torch 2.x
        weights = "DEFAULT" if pretrained else None
        base_model = torchvision.models.resnet18(weights=weights)

        # Remove Average Pooling and FC layers (last 2 items in children)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

    def forward(self, x):
        return self.backbone(x)


class DeconvNeck(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(DeconvNeck, self).__init__()
        layers = []
        current_in = in_channels

        for out_ch in out_channels_list:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_in,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            current_in = out_ch

        self.deconv = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.deconv(x)


class DKN(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, num_classes=Config.NUM_CLASSES):
        super(DKN, self).__init__()

        # 1. Backbone (ResNet18 -> 512 channels, stride 32)
        self.backbone = ResNetBackbone(pretrained=True)

        # 2. Neck (Upsample 32x -> 4x. 512 -> 256 -> 128 -> 64)
        self.neck = DeconvNeck(512, [256, 128, 64])

        head_conv = 64

        # 3. Heads
        # Objectness Head (1 channel)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, Config.HM_CHANNELS, kernel_size=1, stride=1, padding=0
            ),
        )

        # Classification Head (N channels)
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, stride=1, padding=0),
        )

        # Regression Head (4 channels: offset_x, offset_y, log_w, log_h)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, Config.REG_CHANNELS, kernel_size=1, stride=1, padding=0
            ),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize heads
        for head in [self.hm_head, self.cls_head, self.reg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Special initialization for Heatmap Head bias to handle class imbalance
        # -2.19 corresponds to p=0.1 in sigmoid (log(0.1/0.9))
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone
        feats = self.backbone(x)

        # Neck
        feats = self.neck(feats)

        # Heads
        hm = torch.sigmoid(self.hm_head(feats))
        cls_logits = self.cls_head(feats)
        reg = self.reg_head(feats)

        return {"hm": hm, "cls": cls_logits, "reg": reg}


class DKNLoss(nn.Module):
    def __init__(self):
        super(DKNLoss, self).__init__()

    def _focal_loss(self, pred, gt):
        """
        Penalty-reduced focal loss for heatmap.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-12, 1 - 1e-12)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        else:
            return -(pos_loss + neg_loss) / num_pos

    def _cls_loss(self, pred, target, mask):
        """
        Cross Entropy loss at object centers.
        pred: (B, C, H, W)
        target: (B, H, W)
        mask: (B, 1, H, W)
        """
        # Flatten
        b, c, h, w = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.reshape(-1)
        mask = mask.reshape(-1)

        valid_inds = mask > 0
        if valid_inds.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        return F.cross_entropy(pred[valid_inds], target[valid_inds])

    def _reg_loss(self, pred, target, mask):
        """
        L1 loss for regression at object centers.
        """
        mask = mask.expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, outputs, batch):
        hm_pred = outputs["hm"]
        cls_pred = outputs["cls"]
        reg_pred = outputs["reg"]

        hm_target = batch["hm"].to(hm_pred.device)
        cls_target = batch["cls_target"].to(cls_pred.device)
        reg_target = batch["reg_target"].to(reg_pred.device)
        reg_mask = batch["reg_mask"].to(reg_pred.device)

        loss_hm = self._focal_loss(hm_pred, hm_target)
        loss_cls = self._cls_loss(cls_pred, cls_target, reg_mask)
        loss_reg = self._reg_loss(reg_pred, reg_target, reg_mask)

        # Weighted sum (Using 1.0 for all as baseline)
        total_loss = loss_hm + loss_cls + loss_reg

        return total_loss, {
            "loss_hm": loss_hm.item(),
            "loss_cls": loss_cls.item(),
            "loss_reg": loss_reg.item(),
        }
