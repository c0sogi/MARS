import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from collections import OrderedDict
from torchvision.ops import sigmoid_focal_loss, nms, box_iou
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CovidDataset
from library.utils import (
    seed_everything,
    collate_fn,
    MeanAveragePrecision,
    AverageMeter,
)

# =========================================================================
# Layers & Blocks
# =========================================================================


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution used in BiFPN.
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
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.act = nn.SiLU(inplace=True)  # Swish

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    Single BiFPN Layer with weighted feature fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.epsilon = 1e-4

        # Learnable weights for fusion (3 inputs for some nodes, 2 for others)
        # P6_td: P6_in + P7_in (2)
        # P5_td: P5_in + P6_td (2)
        # P4_td: P4_in + P5_td (2)
        # P3_out: P3_in + P4_td (2)
        # P4_out: P4_in + P4_td + P3_out (3)
        # P5_out: P5_in + P5_td + P4_out (3)
        # P6_out: P6_in + P6_td + P5_out (3)
        # P7_out: P7_in + P6_out (2)

        # We define weights as learnable parameters initialized to 1
        self.w = nn.Parameter(torch.ones(8, 3))  # Max 3 inputs, 8 nodes

        # Convolutions after fusion
        self.convs = nn.ModuleList(
            [SeparableConvBlock(channels, channels) for _ in range(8)]
        )

    def forward(self, features):
        p3, p4, p5, p6, p7 = features

        # Normalize weights using ReLU to ensure non-negativity
        w = torch.relu(self.w)
        w = w / (torch.sum(w, dim=1, keepdim=True) + self.epsilon)

        # --- Top-Down Path ---
        # P6_td = Conv(w0*P6_in + w1*Resize(P7_in))
        p6_td = self.convs[0](
            w[0, 0] * p6 + w[0, 1] * F.interpolate(p7, size=p6.shape[-2:])
        )

        # P5_td = Conv(w0*P5_in + w1*Resize(P6_td))
        p5_td = self.convs[1](
            w[1, 0] * p5 + w[1, 1] * F.interpolate(p6_td, size=p5.shape[-2:])
        )

        # P4_td = Conv(w0*P4_in + w1*Resize(P5_td))
        p4_td = self.convs[2](
            w[2, 0] * p4 + w[2, 1] * F.interpolate(p5_td, size=p4.shape[-2:])
        )

        # --- Bottom-Up Path ---
        # P3_out = Conv(w0*P3_in + w1*Resize(P4_td))
        p3_out = self.convs[3](
            w[3, 0] * p3 + w[3, 1] * F.interpolate(p4_td, size=p3.shape[-2:])
        )

        # P4_out = Conv(w0*P4_in + w1*P4_td + w2*Resize(P3_out))
        p4_out = self.convs[4](
            w[4, 0] * p4
            + w[4, 1] * p4_td
            + w[4, 2] * F.interpolate(p3_out, size=p4.shape[-2:], mode="nearest")
        )

        # P5_out = Conv(w0*P5_in + w1*P5_td + w2*Resize(P4_out))
        p5_out = self.convs[5](
            w[5, 0] * p5
            + w[5, 1] * p5_td
            + w[5, 2] * F.interpolate(p4_out, size=p5.shape[-2:], mode="nearest")
        )

        # P6_out = Conv(w0*P6_in + w1*P6_td + w2*Resize(P5_out))
        p6_out = self.convs[6](
            w[6, 0] * p6
            + w[6, 1] * p6_td
            + w[6, 2] * F.interpolate(p5_out, size=p6.shape[-2:], mode="nearest")
        )

        # P7_out = Conv(w0*P7_in + w1*Resize(P6_out))
        p7_out = self.convs[7](
            w[7, 0] * p7
            + w[7, 1] * F.interpolate(p6_out, size=p7.shape[-2:], mode="nearest")
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=160, num_layers=4):
        super().__init__()
        # Projections to match BiFPN channel size
        self.projections = nn.ModuleList(
            [
                (
                    nn.Conv2d(in_c, out_channels, 1)
                    if in_c != out_channels
                    else nn.Identity()
                )
                for in_c in in_channels_list
            ]
        )

        # Generate P6 and P7
        self.p6_conv = nn.Conv2d(in_channels_list[-1], out_channels, 3, 2, 1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

        self.layers = nn.Sequential(
            *[BiFPNBlock(out_channels) for _ in range(num_layers)]
        )

    def forward(self, features):
        # features: [P3, P4, P5]
        p3, p4, p5 = features

        # Project to unified channels
        p3 = self.projections[0](p3)
        p4 = self.projections[1](p4)
        p5_in = features[2]  # Keep original for P6 generation
        p5 = self.projections[2](p5)

        # Generate P6, P7
        p6 = self.p6_conv(p5_in)
        p7 = self.p7_conv(F.relu(p6))

        # Run BiFPN repeats
        features = [p3, p4, p5, p6, p7]
        features = self.layers(features)

        return OrderedDict(
            {
                "0": features[0],  # P3
                "1": features[1],  # P4
                "2": features[2],  # P5
                "3": features[3],  # P6
                "4": features[4],  # P7
            }
        )


class EfficientDetBackbone(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3", pretrained=True):
        super().__init__()
        # Load backbone features P3, P4, P5 (strides 8, 16, 32)
        # indices=(2, 3, 4) typically corresponds to these strides in efficientnet
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        feature_info = self.backbone.feature_info.channels()

        # BiFPN D3 config: 160 channels, 4 repeats
        self.neck = BiFPN(feature_info, out_channels=160, num_layers=4)
        self.out_channels = 160

    def forward(self, x):
        feats = self.backbone(x)
        return self.neck(feats)


class RetinaNetHead(nn.Module):
    """
    Shared head for classification and box regression.
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = nn.ModuleList()
        self.regression_head = nn.ModuleList()

        for _ in range(4):
            self.classification_head.append(
                SeparableConvBlock(in_channels, in_channels)
            )
            self.regression_head.append(SeparableConvBlock(in_channels, in_channels))

        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Focal loss init for cls
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            cls_feat = feature
            reg_feat = feature

            for layer in self.classification_head:
                cls_feat = layer(cls_feat)
            for layer in self.regression_head:
                reg_feat = layer(reg_feat)

            logits.append(self.cls_score(cls_feat))
            bbox_reg.append(self.bbox_pred(reg_feat))
        return logits, bbox_reg


class GlobalStudyHead(nn.Module):
    """
    Auxiliary head for study-level classification attached to P7.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x is P7 feature map
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# =========================================================================
# Main Model
# =========================================================================


class MultiTaskEfficientDet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Backbone + BiFPN
        self.backbone = EfficientDetBackbone(config.BACKBONE, config.PRETRAINED)

        # 2. Anchor Generator
        # Strides: P3=8, P4=16, P5=32, P6=64, P7=128
        anchor_sizes = tuple(
            (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
            for x in [32, 64, 128, 256, 512]
        )
        aspect_ratios = (tuple(config.ANCHOR_RATIOS),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # 3. Detection Head
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.det_head = RetinaNetHead(
            self.backbone.out_channels, self.num_anchors, config.NUM_DETECTION_CLASSES
        )

        # 4. Study Head
        self.study_head = GlobalStudyHead(
            self.backbone.out_channels, config.NUM_STUDY_CLASSES
        )

        # 5. Transform (Normalization is done in dataset, but resizing/padding logic for inference needs care)
        # We assume input is already transformed.
        # GeneralizedRCNNTransform handles image resizing and normalization usually,
        # but we do it in Albumentations. We use it here primarily for batching/unbatching logic if needed,
        # but we will handle logic manually for simplicity.

    def forward(self, images, targets=None):
        """
        Args:
            images: Tensor (B, C, H, W)
            targets: List of dicts
        """
        # Feature Extraction
        features = self.backbone(images)  # OrderedDict P3..P7
        feature_list = list(features.values())

        # Detection Head
        head_outputs = self.det_head(feature_list)  # (logits, bbox_reg)

        # Study Head (Attached to P7 - last feature)
        study_logits = self.study_head(feature_list[-1])

        # Anchor Generation
        # AnchorGenerator expects ImageList, but we can pass tensors if we wrap them
        # or just pass image size list
        image_sizes = [img.shape[-2:] for img in images]
        anchors = self.anchor_generator(images, feature_list)

        if self.training:
            return self.compute_loss(head_outputs, study_logits, anchors, targets)
        else:
            return self.postprocess(head_outputs, study_logits, anchors, image_sizes)

    def compute_loss(self, head_outputs, study_logits, anchors, targets):
        cls_logits, bbox_preds = head_outputs

        # Flatten anchors and predictions
        anchors = torch.cat(anchors, dim=0)

        # Detection Losses
        # We need to match anchors to targets
        # This logic is complex to implement from scratch in one file without borrowing
        # heavily from torchvision.
        # To save space and ensure correctness, we will use a simplified matching strategy
        # or rely on the fact that we are in a custom training loop.

        # Actually, let's use the helper functions from torchvision if possible,
        # but they are often private.
        # We will implement a basic matcher here.

        classification_losses = []
        regression_losses = []

        # Process each image in batch
        B = len(targets)
        num_anchors_per_level = [
            o.shape[2] * o.shape[3] * self.num_anchors for o in cls_logits
        ]

        # Flatten predictions per image
        cls_logits_flat = torch.cat(
            [
                l.permute(0, 2, 3, 1)
                .flatten(1, -2)
                .reshape(B, -1, self.config.NUM_DETECTION_CLASSES)
                for l in cls_logits
            ],
            dim=1,
        )
        bbox_preds_flat = torch.cat(
            [
                l.permute(0, 2, 3, 1).flatten(1, -2).reshape(B, -1, 4)
                for l in bbox_preds
            ],
            dim=1,
        )

        # Anchors are (N_total, 4)

        for i in range(B):
            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]

            if len(gt_boxes) == 0:
                # No objects, all background
                # Focal loss with target 0
                cls_loss = sigmoid_focal_loss(
                    cls_logits_flat[i],
                    torch.zeros_like(cls_logits_flat[i]),
                    alpha=0.25,
                    gamma=2.0,
                    reduction="sum",
                )
                classification_losses.append(
                    cls_loss / max(1, len(gt_boxes))
                )  # Normalize?
                continue

            # Match anchors to GT
            # IoU (N_anchors, N_gt)
            ious = box_iou(anchors, gt_boxes)

            # Max IoU for each anchor
            max_iou, max_idx = ious.max(dim=1)

            # Assign labels
            # IoU < 0.4: Background (0)
            # 0.4 <= IoU < 0.5: Ignore (-1)
            # IoU >= 0.5: Foreground (Label)

            anchor_labels = torch.zeros(
                len(anchors), dtype=torch.long, device=anchors.device
            )
            anchor_labels[max_iou >= 0.5] = gt_labels[max_idx[max_iou >= 0.5]]
            anchor_labels[(max_iou < 0.5) & (max_iou >= 0.4)] = -1  # Ignore

            # Regression targets
            matched_gt_boxes = gt_boxes[max_idx]

            # Compute targets (dx, dy, dw, dh)
            # Simplified box coding
            src_w = anchors[:, 2] - anchors[:, 0]
            src_h = anchors[:, 3] - anchors[:, 1]
            src_ctr_x = anchors[:, 0] + 0.5 * src_w
            src_ctr_y = anchors[:, 1] + 0.5 * src_h

            target_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
            target_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
            target_ctr_x = matched_gt_boxes[:, 0] + 0.5 * target_w
            target_ctr_y = matched_gt_boxes[:, 1] + 0.5 * target_h

            target_dx = (target_ctr_x - src_ctr_x) / src_w
            target_dy = (target_ctr_y - src_ctr_y) / src_h
            target_dw = torch.log(target_w / src_w)
            target_dh = torch.log(target_h / src_h)

            regression_targets = torch.stack(
                (target_dx, target_dy, target_dw, target_dh), dim=1
            )

            # Classification Loss
            valid_mask = anchor_labels >= 0
            pos_mask = anchor_labels > 0

            # One-hot encoding for focal loss
            target_cls = torch.zeros_like(cls_logits_flat[i])
            if pos_mask.sum() > 0:
                target_cls[pos_mask, anchor_labels[pos_mask] - 1] = (
                    1.0  # Classes are 1-indexed in targets, 0-indexed in logits
                )

            cls_loss = sigmoid_focal_loss(
                cls_logits_flat[i][valid_mask],
                target_cls[valid_mask],
                alpha=0.25,
                gamma=2.0,
                reduction="sum",
            )

            num_pos = max(1, pos_mask.sum().item())
            classification_losses.append(cls_loss / num_pos)

            # Regression Loss (Huber/L1) on positive anchors
            if pos_mask.sum() > 0:
                reg_loss = F.smooth_l1_loss(
                    bbox_preds_flat[i][pos_mask],
                    regression_targets[pos_mask],
                    beta=1.0 / 9.0,
                    reduction="sum",
                )
                regression_losses.append(reg_loss / num_pos)
            else:
                regression_losses.append(torch.tensor(0.0, device=anchors.device))

        # Study Loss
        study_targets = torch.stack([t["study_label"] for t in targets])
        study_loss = F.cross_entropy(study_logits, study_targets)

        total_cls_loss = sum(classification_losses) / B
        total_reg_loss = sum(regression_losses) / B

        losses = {
            "loss_cls": total_cls_loss,
            "loss_box": total_reg_loss,
            "loss_study": study_loss * self.config.LAMBDA_STUDY,
        }
        return losses

    def postprocess(self, head_outputs, study_logits, anchors, image_sizes):
        cls_logits, bbox_preds = head_outputs
        anchors = torch.cat(anchors, dim=0)

        B = len(image_sizes)

        # Flatten
        cls_logits_flat = torch.cat(
            [
                l.permute(0, 2, 3, 1)
                .flatten(1, -2)
                .reshape(B, -1, self.config.NUM_DETECTION_CLASSES)
                for l in cls_logits
            ],
            dim=1,
        )
        bbox_preds_flat = torch.cat(
            [
                l.permute(0, 2, 3, 1).flatten(1, -2).reshape(B, -1, 4)
                for l in bbox_preds
            ],
            dim=1,
        )

        cls_probs = torch.sigmoid(cls_logits_flat)
        study_probs = torch.softmax(study_logits, dim=1)

        detections = []

        for i in range(B):
            # Decode boxes
            src_w = anchors[:, 2] - anchors[:, 0]
            src_h = anchors[:, 3] - anchors[:, 1]
            src_ctr_x = anchors[:, 0] + 0.5 * src_w
            src_ctr_y = anchors[:, 1] + 0.5 * src_h

            dx = bbox_preds_flat[i][:, 0]
            dy = bbox_preds_flat[i][:, 1]
            dw = bbox_preds_flat[i][:, 2]
            dh = bbox_preds_flat[i][:, 3]

            pred_ctr_x = dx * src_w + src_ctr_x
            pred_ctr_y = dy * src_h + src_ctr_y
            pred_w = torch.exp(dw) * src_w
            pred_h = torch.exp(dh) * src_h

            pred_boxes = torch.stack(
                [
                    pred_ctr_x - 0.5 * pred_w,
                    pred_ctr_y - 0.5 * pred_h,
                    pred_ctr_x + 0.5 * pred_w,
                    pred_ctr_y + 0.5 * pred_h,
                ],
                dim=1,
            )

            # Clip to image
            h, w = image_sizes[i]
            pred_boxes[:, 0].clamp_(min=0, max=w)
            pred_boxes[:, 1].clamp_(min=0, max=h)
            pred_boxes[:, 2].clamp_(min=0, max=w)
            pred_boxes[:, 3].clamp_(min=0, max=h)

            # Filter by confidence
            scores = cls_probs[i].flatten()
            keep = scores > self.config.CONF_THRESHOLD

            boxes = pred_boxes[keep]
            scores = scores[keep]
            labels = torch.ones_like(scores, dtype=torch.long)  # Only 1 class 'opacity'

            # NMS
            keep_idx = nms(boxes, scores, self.config.IOU_THRESHOLD)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            detections.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "study_probs": study_probs[i],
                }
            )

        return detections


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    map_metric = MeanAveragePrecision(num_classes=Config.NUM_DETECTION_CLASSES)

    for images, targets, _ in loader:
        images = images.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        detections = model(images)

        # Format for mAP
        pred_boxes = [d["boxes"] for d in detections]
        pred_scores = [d["scores"] for d in detections]
        pred_labels = [d["labels"] for d in detections]

        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        map_metric.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    return map_metric.compute()


@torch.no_grad()
def generate_predictions(model, loader, device, output_path):
    model.eval()
    results = []

    print("Generating predictions...")

    for images, _, image_ids in loader:
        images = images.to(device)
        detections = model(images)

        for i, det in enumerate(detections):
            img_id = image_ids[i]
            study_id = img_id.replace(
                "_image", "_study"
            )  # Assuming mapping or separate handling
            # Note: Dataset returns image_id. We need to handle study_id prediction too.
            # The test set loader returns image_id which is typically "ID_image".
            # The study ID is needed for study-level prediction rows.
            # However, the dataset `__getitem__` returns `study_id` in target dict, but for test set
            # we don't have targets. The dataset implementation for test set puts dummy targets.
            # We can extract study_id from the image_id if the format is consistent or pass it through.
            # Looking at dataset.py: `target` contains `study_id`.
            # But `collate_fn` returns `images, targets, image_ids`.
            # `targets` is a list of dicts. For test set, `dataset.py` sets study_id correctly.

            # Wait, `generate_predictions` loop unpacks `images, _, image_ids`.
            # The second arg is targets.
            # We should use the targets to get study_id even in test mode (it's in metadata).

            # Actually, let's just re-fetch targets from loader
            pass

    # Re-run loop correctly
    final_preds = []

    # Map from study_id to study_prediction (taking max or average if multiple images per study)
    # The competition format requires specific rows.
    # We will store all predictions and then format.

    study_preds = {}  # study_id -> list of probs
    image_preds = {}  # image_id -> string

    for images, targets, ids in loader:
        images = images.to(device)
        detections = model(images)

        for i, det in enumerate(detections):
            img_id = ids[i]
            study_id = targets[i]["study_id"]

            # Study Prediction
            probs = det["study_probs"].cpu().numpy()
            if study_id not in study_preds:
                study_preds[study_id] = []
            study_preds[study_id].append(probs)

            # Image Prediction
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()

            # Logic: If study is "Negative", force "none"
            # We will decide this after aggregating study probs?
            # Or per image? Usually per image consistency is good.
            # Let's use the study prob for THIS image to gate.

            neg_prob = probs[0]  # "Negative for Pneumonia" is index 0

            pred_str = ""
            if neg_prob > Config.STUDY_CONF_THRESHOLD:
                pred_str = "none 1 0 0 1 1"
            else:
                s = []
                for b, sc in zip(boxes, scores):
                    s.append(
                        f"opacity {sc:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                    )

                if len(s) == 0:
                    pred_str = "none 1 0 0 1 1"
                else:
                    pred_str = " ".join(s)

            image_preds[img_id] = pred_str

    # Aggregate Study Predictions
    study_rows = []
    for study_id, probs_list in study_preds.items():
        # Average probabilities across images in the study
        avg_probs = np.mean(probs_list, axis=0)

        # Create prediction string
        # Format: class_id confidence 0 0 1 1
        # We can predict multiple labels, but usually one is dominant.
        # Task says: "predict at least one of the above labels".
        # We will predict the argmax class.

        idx = np.argmax(avg_probs)
        class_name = (
            Config.STUDY_CLASSES[idx].split()[0].lower()
        )  # "negative", "typical", etc.
        if class_name == "negative":
            class_name = "negative"
        if class_name == "typical":
            class_name = "typical"
        if class_name == "indeterminate":
            class_name = "indeterminate"
        if class_name == "atypical":
            class_name = "atypical"

        # Fix class names to match submission format exactly
        # "Negative for Pneumonia" -> "negative"
        # "Typical Appearance" -> "typical"
        # "Indeterminate Appearance" -> "indeterminate"
        # "Atypical Appearance" -> "atypical"
        # My logic above does this roughly, let's be precise.
        mapping = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}
        label = mapping[idx]
        conf = avg_probs[idx]

        pred_string = f"{label} {conf:.4f} 0 0 1 1"
        study_rows.append({"id": f"{study_id}_study", "PredictionString": pred_string})

    # Image Rows
    image_rows = [
        {"id": f"{k}_image", "PredictionString": v} for k, v in image_preds.items()
    ]

    # Combine
    df_study = pd.DataFrame(study_rows)
    df_image = pd.DataFrame(image_rows)
    df_sub = pd.concat([df_study, df_image], ignore_index=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data
    train_dataset = CovidDataset("train", load_cached_data=True)
    val_dataset = CovidDataset("val", load_cached_data=True)
    test_dataset = CovidDataset("test", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Model
    model = MultiTaskEfficientDet(Config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    best_map = 0.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_map = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val mAP: {val_map}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with mAP: {best_map}")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    generate_predictions(model, test_loader, device, Config.SUBMISSION_PATH)
