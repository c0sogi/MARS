import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from library.config import Config
from library.utils import calculate_map, get_bbox_from_mask


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()

    # Meters
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    # Loss Functions
    # Study: Multi-class classification
    criterion_cls = nn.CrossEntropyLoss()
    # Segmentation: Binary classification per pixel
    criterion_seg = nn.BCEWithLogitsLoss()

    start_time = time.time()

    for step, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Outputs: logit_cls, logit_seg
        logit_cls, logit_seg = model(images)

        # -------------------------
        # 1. Classification Loss
        # -------------------------
        # labels are one-hot (N, 4), CrossEntropy expects class indices (N,) or probabilities
        # Since we have one-hot, we take argmax for target indices
        target_cls = torch.argmax(labels, dim=1)
        loss_cls = criterion_cls(logit_cls, target_cls)

        # -------------------------
        # 2. Segmentation Loss
        # -------------------------
        loss_seg = criterion_seg(logit_seg, masks)

        # -------------------------
        # 3. Total Loss
        # -------------------------
        loss_total = (
            Config.CLS_LOSS_WEIGHT * loss_cls + Config.SEG_LOSS_WEIGHT * loss_seg
        )

        # Backward
        loss_total.backward()

        # Optimizer Step
        optimizer.step()

        # Update Meters
        loss_meter.update(loss_total.item(), batch_size)
        cls_loss_meter.update(loss_cls.item(), batch_size)
        seg_loss_meter.update(loss_seg.item(), batch_size)

    # Note: Scheduler step is typically handled after the epoch in the main loop
    # or per iteration if it's OneCycleLR. Assuming CosineAnnealingLR (epoch-based)
    # handled by caller.

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch} Train | "
        f"Loss: {loss_meter.avg:.6f} | "
        f"Cls Loss: {cls_loss_meter.avg:.6f} | "
        f"Seg Loss: {seg_loss_meter.avg:.6f} | "
        f"Time: {elapsed:.2f}s"
    )

    return loss_meter.avg


def valid_one_epoch(model, dataloader, device, epoch):
    """
    Performs one epoch of validation.
    Calculates Loss, Study Accuracy, and Box mAP.
    """
    model.eval()

    loss_meter = AverageMeter()

    # For Metrics
    pred_boxes_all = []
    pred_scores_all = []
    gt_boxes_all = []

    correct_studies = 0
    total_studies = 0

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    start_time = time.time()

    with torch.no_grad():
        for step, (images, masks, labels) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # Forward
            logit_cls, logit_seg = model(images)

            # -------------------------
            # Loss Calculation (for monitoring)
            # -------------------------
            target_cls = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(logit_cls, target_cls)

            loss_seg = criterion_seg(logit_seg, masks)

            loss_total = (
                Config.CLS_LOSS_WEIGHT * loss_cls + Config.SEG_LOSS_WEIGHT * loss_seg
            )

            loss_meter.update(loss_total.item(), batch_size)

            # -------------------------
            # Study Accuracy
            # -------------------------
            preds_cls = torch.argmax(logit_cls, dim=1)
            correct_studies += (preds_cls == target_cls).sum().item()
            total_studies += batch_size

            # -------------------------
            # Image mAP Prep
            # -------------------------
            # Sigmoid on final mask
            probs_seg = torch.sigmoid(logit_seg)  # (B, 1, H, W)

            # Convert to CPU numpy for box extraction
            probs_seg_np = probs_seg.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            for i in range(batch_size):
                # Prediction
                prob_map = probs_seg_np[i, 0]
                # Threshold at 0.5 for binary mask to get contours
                binary_mask = (prob_map > 0.5).astype(np.uint8)

                # Extract boxes
                p_boxes = get_bbox_from_mask(binary_mask)

                # Calculate scores for each box (mean probability inside the box)
                p_scores = []
                for box in p_boxes:
                    x1, y1, x2, y2 = box
                    # Clip coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(prob_map.shape[1], x2)
                    y2 = min(prob_map.shape[0], y2)

                    if x2 > x1 and y2 > y1:
                        score = np.mean(prob_map[y1:y2, x1:x2])
                    else:
                        score = 0.0
                    p_scores.append(score)

                pred_boxes_all.append(p_boxes)
                pred_scores_all.append(p_scores)

                # Ground Truth
                # We need to extract boxes from the GT mask for mAP calculation
                gt_mask_i = masks_np[i, 0]
                g_boxes = get_bbox_from_mask(gt_mask_i)
                gt_boxes_all.append(g_boxes)

    # Calculate Metrics
    study_acc = correct_studies / total_studies if total_studies > 0 else 0.0

    # Calculate mAP (IoU > 0.5)
    map_score = calculate_map(
        pred_boxes_all, pred_scores_all, gt_boxes_all, iou_threshold=0.5
    )

    # Composite Score (Average of Study Acc and Box mAP)
    # This aligns with the strategy to balance both tasks
    composite_score = (study_acc + map_score) / 2.0

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch} Valid | "
        f"Loss: {loss_meter.avg:.10f} | "
        f"Study Acc: {study_acc:.10f} | "
        f"Box mAP: {map_score:.10f} | "
        f"Composite: {composite_score:.10f} | "
        f"Time: {elapsed:.2f}s"
    )

    return {
        "loss": loss_meter.avg,
        "study_acc": study_acc,
        "map_score": map_score,
        "composite_score": composite_score,
    }
