import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from library.config import Config


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Net (PFN).
    Converts pillar inputs (N, P, D) into pillar features (N, C).
    """

    def __init__(self, num_input_features=9, num_output_features=64):
        super().__init__()
        self.linear = nn.Linear(num_input_features, num_output_features, bias=False)
        self.bn = nn.BatchNorm1d(num_output_features)

    def forward(self, pillars, n_points):
        """
        Args:
            pillars: (N, P, D) tensor of point features.
            n_points: (N,) tensor of number of points per pillar (unused in simplified PFN).
        Returns:
            (N, C) tensor of pillar features.
        """
        # Linear transformation: (N, P, D) -> (N, P, C)
        x = self.linear(pillars)

        # Permute for BatchNorm: (N, C, P)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = F.relu(x)

        # Max pooling over points: (N, C, P) -> (N, C)
        x = torch.max(x, dim=2)[0]
        return x


class PointPillarsScatter(nn.Module):
    """
    Scatters pillar features into a 2D pseudo-image.
    """

    def __init__(self, num_features=64, grid_size=Config.GRID_SIZE):
        super().__init__()
        self.num_features = num_features
        self.nx, self.ny, _ = grid_size

    def forward(self, pillar_features, coors, batch_size):
        """
        Args:
            pillar_features: (N, C) tensor.
            coors: (N, 4) tensor of [batch_idx, z, y, x].
            batch_size: int.
        Returns:
            (B, C, H, W) tensor.
        """
        # Create empty canvas
        canvas = torch.zeros(
            (batch_size, self.num_features, self.ny, self.nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        # Unpack coordinates
        batch_idx = coors[:, 0]
        y_idx = coors[:, 2]
        x_idx = coors[:, 3]

        # Scatter features
        canvas[batch_idx, :, y_idx, x_idx] = pillar_features

        return canvas


class Backbone(nn.Module):
    """
    2D CNN Backbone for PointPillars.
    Downsamples the pseudo-image and then upsamples/concatenates features.
    Designed for a stride of 2 relative to the input grid.
    """

    def __init__(self, input_channels=64):
        super().__init__()

        # Block 1: Stride 2 (512x512 -> 256x256)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Block 2: Stride 2 (256x256 -> 128x128)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Block 3: Stride 2 (128x128 -> 64x64)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Upsampling Layers to 256x256
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, 3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 64, 512, 512)
        x1 = self.block1(x)  # (B, 64, 256, 256)
        x2 = self.block2(x1)  # (B, 128, 128, 128)
        x3 = self.block3(x2)  # (B, 256, 64, 64)

        u1 = self.up1(x1)  # (B, 128, 256, 256)
        u2 = self.up2(x2)  # (B, 128, 256, 256)
        u3 = self.up3(x3)  # (B, 128, 256, 256)

        out = torch.cat([u1, u2, u3], dim=1)  # (B, 384, 256, 256)
        return out


class SSDHead(nn.Module):
    """
    Single Shot Detector Head.
    Predicts classification scores and regression offsets.
    """

    def __init__(self, input_channels=384, num_anchors=18):
        super().__init__()
        # Classification: 1 score per anchor (Binary: Object vs BG for that specific anchor class)
        self.cls_head = nn.Conv2d(input_channels, num_anchors, 1)

        # Regression: 7 targets per anchor (x, y, z, w, l, h, theta)
        self.reg_head = nn.Conv2d(input_channels, num_anchors * 7, 1)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - pi) / pi))
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.constant_(self.reg_head.bias, 0)

    def forward(self, x):
        cls_score = self.cls_head(x)
        reg_pred = self.reg_head(x)
        return cls_score, reg_pred


class PointPillars(nn.Module):
    """
    End-to-End PointPillars Network.
    """

    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet(
            num_input_features=9, num_output_features=Config.NUM_PILLAR_FEATURES
        )
        self.scatter = PointPillarsScatter(num_features=Config.NUM_PILLAR_FEATURES)
        self.backbone = Backbone(input_channels=Config.NUM_PILLAR_FEATURES)

        # Anchors: 9 Classes * 2 Rotations = 18 anchors per grid location
        num_anchors = len(Config.CLASS_NAMES) * len(Config.ANCHOR_ROTATIONS)
        self.head = SSDHead(num_anchors=num_anchors)

    def forward(self, data_dict):
        pillars = data_dict["pillars"]
        coors = data_dict["coors"]
        n_points = data_dict["n_points"]
        sample_tokens = data_dict.get("sample_tokens", [])

        batch_size = len(sample_tokens)

        # 1. Feature Encoding
        pillar_features = self.pfn(pillars, n_points)

        # 2. Scatter to Grid
        spatial_features = self.scatter(pillar_features, coors, batch_size)

        # 3. Backbone
        neck_features = self.backbone(spatial_features)

        # 4. Detection Head
        cls_preds, reg_preds = self.head(neck_features)

        output = {"cls_preds": cls_preds, "reg_preds": reg_preds}

        # 5. Loss Calculation (during training)
        if self.training and "cls_targets" in data_dict:
            loss_dict = self.loss(
                cls_preds, reg_preds, data_dict["cls_targets"], data_dict["reg_targets"]
            )
            output.update(loss_dict)

        return output

    def loss(self, cls_preds, reg_preds, cls_targets, reg_targets):
        """
        Calculates Focal Loss for classification and Smooth L1 Loss for regression.
        """
        # Flatten predictions to match targets
        # cls_preds: (B, 18, H, W) -> (B, H, W, 18) -> (-1)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous().view(-1)
        # reg_preds: (B, 18*7, H, W) -> (B, H, W, 18, 7) -> (-1, 7)
        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous().view(-1, 7)

        cls_targets = cls_targets.view(-1)
        reg_targets = reg_targets.view(-1, 7)

        # Create masks
        pos_mask = cls_targets > 0
        # neg_mask = cls_targets == 0
        valid_mask = cls_targets != -1

        # --- Classification Loss (Sigmoid Focal Loss) ---
        # Prepare binary targets (1 for object, 0 for background)
        labels = torch.zeros_like(cls_preds)
        labels[pos_mask] = 1.0

        alpha = Config.FOCAL_ALPHA
        gamma = Config.FOCAL_GAMMA

        probs = torch.sigmoid(cls_preds)
        ce_loss = F.binary_cross_entropy_with_logits(
            cls_preds, labels, reduction="none"
        )
        p_t = probs * labels + (1 - probs) * (1 - labels)
        loss_cls = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
            loss_cls = alpha_t * loss_cls

        # Normalize by number of positive samples
        num_pos = pos_mask.sum().float()
        loss_cls = loss_cls[valid_mask].sum() / (num_pos + 1e-6)

        # --- Regression Loss (Smooth L1) ---
        loc_loss = F.smooth_l1_loss(
            reg_preds[pos_mask], reg_targets[pos_mask], reduction="sum", beta=1.0 / 9.0
        )
        loc_loss = loc_loss / (num_pos + 1e-6)

        # Weighted Sum
        total_loss = (
            Config.LOSS_WEIGHTS["cls_weight"] * loss_cls
            + Config.LOSS_WEIGHTS["loc_weight"] * loc_loss
        )

        return {"loss": total_loss, "cls_loss": loss_cls, "loc_loss": loc_loss}
