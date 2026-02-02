import torch
import numpy as np
import sys
from library.utils import AverageMeter, get_boxes_from_mask, calculate_map
from library.config import Config


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    cls_losses = AverageMeter()
    seg_losses = AverageMeter()
    dice_losses = AverageMeter()

    # Iterate over data
    for i, (images, labels, masks) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Forward pass
        cls_logits, seg_logits = model(images)

        # Calculate loss
        loss, cls_loss, seg_loss, dice_loss = criterion(
            cls_logits, seg_logits, labels, masks
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        seg_losses.update(seg_loss.item(), batch_size)
        dice_losses.update(dice_loss.item(), batch_size)

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{Config.EPOCHS}] Train Summary:")
    print(f"  Total Loss: {losses.avg}")
    print(f"  Cls Loss:   {cls_losses.avg}")
    print(f"  Seg Loss:   {seg_losses.avg}")
    print(f"  Dice Loss:  {dice_losses.avg}")

    return losses.avg


def evaluate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set and calculates mAP.
    """
    model.eval()

    losses = AverageMeter()
    cls_losses = AverageMeter()
    seg_losses = AverageMeter()

    # Containers for mAP calculation
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_ids = []
    all_gt_boxes = []
    all_gt_ids = []

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            cls_logits, seg_logits = model(images)

            # Calculate loss
            loss, cls_loss, seg_loss, _ = criterion(
                cls_logits, seg_logits, labels, masks
            )

            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            cls_losses.update(cls_loss.item(), batch_size)
            seg_losses.update(seg_loss.item(), batch_size)

            # --- Prepare for mAP Calculation ---
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(seg_logits)

            # Convert to numpy for box extraction
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(batch_size):
                # 1. Get Predicted Boxes
                # probs_np shape is (B, 1, H, W), taking [i, 0] gives (H, W)
                pred_mask = probs_np[i, 0]
                p_boxes = get_boxes_from_mask(pred_mask, threshold=0.5)

                # Calculate confidence for each box (mean pixel probability within box)
                p_scores = []
                for box in p_boxes:
                    # box: [xmin, ymin, xmax, ymax]
                    x1, y1, x2, y2 = box
                    # Ensure indices are within bounds
                    y1 = max(0, y1)
                    x1 = max(0, x1)
                    y2 = min(pred_mask.shape[0], y2)
                    x2 = min(pred_mask.shape[1], x2)

                    roi = pred_mask[y1:y2, x1:x2]
                    if roi.size > 0:
                        p_scores.append(np.mean(roi))
                    else:
                        p_scores.append(0.0)

                # Class ID for opacity is 0 (we only detect one object class: opacity)
                p_ids = [0] * len(p_boxes)

                all_pred_boxes.append(p_boxes)
                all_pred_scores.append(p_scores)
                all_pred_ids.append(p_ids)

                # 2. Get Ground Truth Boxes
                # We extract boxes from the GT mask as that's what the loader provides
                gt_mask = masks_np[i, 0]
                g_boxes = get_boxes_from_mask(gt_mask, threshold=0.5)
                g_ids = [0] * len(g_boxes)

                all_gt_boxes.append(g_boxes)
                all_gt_ids.append(g_ids)

    # Calculate mAP
    map_score = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_pred_ids,
        all_gt_boxes,
        all_gt_ids,
        iou_threshold=0.5,
    )

    print(f"Validation Summary:")
    print(f"  Total Loss: {losses.avg}")
    print(f"  Cls Loss:   {cls_losses.avg}")
    print(f"  Seg Loss:   {seg_losses.avg}")
    print(f"  mAP @ 0.5:  {map_score}")

    return losses.avg, map_score
