import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class BevBackbone(nn.Module):
    """
    Backbone network for extracting features from BEV images.
    Uses ResNet-18 layers and a simple FPN-like upsampling structure
    to output features at Stride 4 (128x128).
    """

    def __init__(self, in_channels=3):
        super(BevBackbone, self).__init__()

        # Load pre-trained ResNet18
        # Note: We ignore the 'pretrained' warning for this implementation
        # as we want to allow offline usage if weights are cached,
        # or standard initialization if not.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adjust first layer if input channels differ from RGB (3)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Extract layers
        # C1: Stride 4 (after maxpool)
        self.layer0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1  # Stride 4, 64 ch
        self.layer2 = self.resnet.layer2  # Stride 8, 128 ch
        self.layer3 = self.resnet.layer3  # Stride 16, 256 ch

        # Lateral connections to reduce/normalize channels before fusion
        self.lat1 = nn.Conv2d(64, 128, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 128, kernel_size=1)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, 512, 512)

        # Forward pass through ResNet
        x = self.layer0(x)
        c1 = self.layer1(x)  # (B, 64, 128, 128)
        c2 = self.layer2(c1)  # (B, 128, 64, 64)
        c3 = self.layer3(c2)  # (B, 256, 32, 32)

        # Upsampling and Fusion
        # Target size is c1 size (128x128)

        p1 = self.lat1(c1)  # (B, 128, 128, 128)

        p2 = self.lat2(c2)  # (B, 128, 64, 64)
        p2 = F.interpolate(p2, size=p1.shape[2:], mode="bilinear", align_corners=False)

        p3 = self.lat3(c3)  # (B, 128, 32, 32)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate along channel dimension
        out = torch.cat([p1, p2, p3], dim=1)  # (B, 384, 128, 128)

        # Fuse
        out = self.fusion(out)  # (B, 128, 128, 128)

        return out


class BevYolo(nn.Module):
    """
    Main 3D Object Detection Model.
    """

    def __init__(self):
        super(BevYolo, self).__init__()

        self.backbone = BevBackbone(in_channels=Config.IN_CHANNELS)

        self.num_anchors = len(Config.ANCHORS)
        self.num_classes = Config.NUM_CLASSES

        # Output Attributes per Anchor:
        # 1 (Objectness) + 8 (Regression) + Num_Classes (Classification)
        # Regression: dx, dy, dw, dl, z, dh, sin, cos
        self.num_attrib = 1 + 8 + self.num_classes

        # Detection Head
        self.head = nn.Conv2d(128, self.num_anchors * self.num_attrib, kernel_size=1)

    def forward(self, x):
        # Backbone features
        feat = self.backbone(x)  # (B, 128, 128, 128)

        # Head
        out = self.head(feat)  # (B, A*Attrib, H, W)

        # Reshape to separate anchors and attributes
        B, C, H, W = out.shape
        out = out.view(B, self.num_anchors, self.num_attrib, H, W)

        # Permute to (B, A, H, W, Attrib) for easier loss calculation/decoding
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        return out


class YoloLoss(nn.Module):
    """
    Loss function for the BevYolo model.
    Computes Objectness Loss, Regression Loss, and Classification Loss.
    """

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, A, H, W, 9 + Num_Classes)
                [obj, dx, dy, dw, dl, z, dh, sin, cos, class_logits...]
            targets: (B, A, H, W, 10)
                [valid, dx, dy, dw, dl, z, dh, sin, cos, class_idx]

        Returns:
            total_loss: Scalar tensor
            metrics: Dict of individual loss components
        """
        # 1. Unpack Targets
        # valid flag indicates if an object is assigned to this anchor/cell
        mask_obj = targets[..., 0] == 1

        target_box = targets[..., 1:9]
        target_cls = targets[..., 9].long()

        # 2. Unpack Predictions
        pred_obj = predictions[..., 0]
        pred_box = predictions[..., 1:9]
        pred_cls = predictions[..., 9:]

        # 3. Objectness Loss (BCE)
        # Applied to ALL cells
        loss_obj = self.bce_loss(pred_obj, targets[..., 0])
        loss_obj = loss_obj.mean()

        # 4. Regression & Classification Loss
        # Applied ONLY to positive cells (mask_obj)
        if mask_obj.sum() > 0:
            pred_box_masked = pred_box[mask_obj]
            target_box_masked = target_box[mask_obj]

            loss_reg = self.mse_loss(pred_box_masked, target_box_masked)
            loss_reg = loss_reg.mean()

            pred_cls_masked = pred_cls[mask_obj]
            target_cls_masked = target_cls[mask_obj]

            loss_cls = self.ce_loss(pred_cls_masked, target_cls_masked)
            loss_cls = loss_cls.mean()
        else:
            loss_reg = torch.tensor(0.0, device=predictions.device)
            loss_cls = torch.tensor(0.0, device=predictions.device)

        # 5. Total Loss
        # Weights can be tuned.
        # Usually regression needs higher weight to converge well on localization.
        w_obj = 1.0
        w_reg = 2.0
        w_cls = 1.0

        total_loss = (w_obj * loss_obj) + (w_reg * loss_reg) + (w_cls * loss_cls)

        return total_loss, {
            "loss_obj": loss_obj.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
        }
