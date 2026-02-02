import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, mask_to_boxes, calculate_map


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for binary segmentation.
    Loss = 1 - (2 * Intersection + Smooth) / (Union + Smooth)
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        data_loader: The training data loader.
        device: The device to train on.
        epoch: Current epoch number.

    Returns:
        dict: A dictionary containing average losses for the epoch.
    """
    model.train()

    # Loss meters
    total_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    # Loss functions
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    for i, batch in enumerate(data_loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        # Forward pass
        seg_logits, cls_logits = model(images)

        # --- Compute Losses ---

        # 1. Segmentation Loss (Hybrid: BCE + Dice)
        seg_bce = bce_loss_fn(seg_logits, masks)
        seg_dice = dice_loss_fn(seg_logits, masks)
        seg_loss = (Config.SEG_LOSS_BCE_WEIGHT * seg_bce) + (
            Config.SEG_LOSS_DICE_WEIGHT * seg_dice
        )

        # 2. Classification Loss (BCE)
        cls_loss = bce_loss_fn(cls_logits, labels)

        # 3. Total Loss
        loss = (Config.TOTAL_LOSS_SEG_WEIGHT * seg_loss) + (
            Config.TOTAL_LOSS_CLASS_WEIGHT * cls_loss
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update meters
        total_loss_meter.update(loss.item(), batch_size)
        seg_loss_meter.update(seg_loss.item(), batch_size)
        cls_loss_meter.update(cls_loss.item(), batch_size)

    # Print metrics
    print(
        f"Epoch [{epoch}] Train Loss: {total_loss_meter.avg:.6f} "
        f"(Seg: {seg_loss_meter.avg:.6f}, Cls: {cls_loss_meter.avg:.6f})"
    )

    return {
        "loss": total_loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
    }


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set and computes mAP.

    Args:
        model: The PyTorch model.
        data_loader: The validation data loader.
        device: The device to evaluate on.

    Returns:
        dict: A dictionary containing average losses and mAP.
    """
    model.eval()

    # Loss meters
    total_loss_meter = AverageMeter()

    # Loss functions for validation tracking
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    # Storage for mAP calculation
    all_pred_boxes = []
    all_true_boxes = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            batch_size = images.size(0)

            # Forward pass
            seg_logits, cls_logits = model(images)

            # Compute Loss (for monitoring)
            seg_bce = bce_loss_fn(seg_logits, masks)
            seg_dice = dice_loss_fn(seg_logits, masks)
            seg_loss = (Config.SEG_LOSS_BCE_WEIGHT * seg_bce) + (
                Config.SEG_LOSS_DICE_WEIGHT * seg_dice
            )
            cls_loss = bce_loss_fn(cls_logits, labels)
            loss = (Config.TOTAL_LOSS_SEG_WEIGHT * seg_loss) + (
                Config.TOTAL_LOSS_CLASS_WEIGHT * cls_loss
            )

            total_loss_meter.update(loss.item(), batch_size)

            # --- Prepare for mAP Calculation ---

            # Convert logits to probabilities
            seg_probs = torch.sigmoid(seg_logits)

            # Move data to CPU for post-processing
            seg_probs_np = seg_probs.cpu().numpy()  # Shape: (B, 1, H, W)
            masks_np = masks.cpu().numpy()  # Shape: (B, 1, H, W)

            for i in range(batch_size):
                # 1. Process Predictions
                # Squeeze channel dim: (1, H, W) -> (H, W)
                curr_prob_map = seg_probs_np[i, 0]

                # Get boxes from probability map
                pred_boxes_list = mask_to_boxes(
                    curr_prob_map, threshold=Config.PIXEL_THRESHOLD
                )

                # Calculate scores for each box (mean probability inside the box)
                scores = []
                for box in pred_boxes_list:
                    x1, y1, x2, y2 = box
                    # Ensure indices are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(curr_prob_map.shape[1], x2), min(
                        curr_prob_map.shape[0], y2
                    )

                    if x2 > x1 and y2 > y1:
                        box_score = np.mean(curr_prob_map[y1:y2, x1:x2])
                    else:
                        box_score = 0.0
                    scores.append(box_score)

                all_pred_boxes.append({"boxes": pred_boxes_list, "scores": scores})

                # 2. Process Ground Truth
                # Squeeze channel dim
                curr_gt_mask = masks_np[i, 0]

                # Reconstruct GT boxes from mask
                # Using 0.5 threshold since GT mask is binary (0 or 1)
                gt_boxes_list = mask_to_boxes(curr_gt_mask, threshold=0.5)

                all_true_boxes.append({"boxes": gt_boxes_list})

    # Compute mAP
    # We use the utility function provided in library/utils.py
    map_score = calculate_map(all_pred_boxes, all_true_boxes, iou_threshold=0.5)

    print(f"Validation Loss: {total_loss_meter.avg:.6f} | mAP@0.5: {map_score:.6f}")

    return {"loss": total_loss_meter.avg, "map": map_score}
