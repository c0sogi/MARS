import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
from library.config import Config
from library.loss import HybridLoss
from library.utils import mask2box, calculate_map


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (DataLoader): Training data loader.
        device (torch.device): Compute device.
        epoch (int): Current epoch number.

    Returns:
        float: Average total loss for the epoch.
    """
    model.train()
    criterion = HybridLoss()

    total_loss_meter = 0.0
    study_loss_meter = 0.0
    seg_loss_meter = 0.0
    count = 0

    # Iterate over data loader
    # Using simple iteration without tqdm to keep logs clean as requested,
    # or minimal print statements.

    for batch_idx, (images, masks, labels) in enumerate(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns study_logits and a list of mask_logits (Deep Supervision)
        study_logits, mask_logits_list = model(images)

        # Calculate loss
        loss_dict = criterion(study_logits, mask_logits_list, labels, masks)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        total_loss_meter += loss.item() * batch_size
        study_loss_meter += loss_dict["study_loss"].item() * batch_size
        seg_loss_meter += loss_dict["seg_loss"].item() * batch_size
        count += batch_size

    avg_loss = total_loss_meter / count
    avg_study_loss = study_loss_meter / count
    avg_seg_loss = seg_loss_meter / count

    print(
        f"Epoch [{epoch}] Train Loss: {avg_loss:.5f} (Study: {avg_study_loss:.5f}, Seg: {avg_seg_loss:.5f})"
    )

    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and mAP.

    Args:
        model (nn.Module): The PyTorch model.
        data_loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        tuple: (average_loss, mAP_score)
    """
    model.eval()
    criterion = HybridLoss()

    total_loss_meter = 0.0
    count = 0

    # Lists for mAP calculation
    all_pred_boxes = []
    all_pred_scores = []
    all_true_boxes = []

    with torch.no_grad():
        for images, masks, labels in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # Forward pass
            # In eval mode, model returns study_logits and single mask_logits (final resolution)
            study_logits, mask_logits = model(images)

            # Calculate validation loss for monitoring
            loss_dict = criterion(study_logits, mask_logits, labels, masks)
            total_loss_meter += loss_dict["loss"].item() * batch_size
            count += batch_size

            # --- Prepare Predictions for mAP ---

            # 1. Study Prediction
            study_probs = torch.softmax(study_logits, dim=1)
            study_preds = torch.argmax(study_probs, dim=1).cpu().numpy()

            # 2. Mask Prediction
            mask_probs = torch.sigmoid(mask_logits)
            mask_preds_np = mask_probs.cpu().numpy()  # (B, 1, H, W)

            # 3. Ground Truth Boxes
            # Convert GT masks to boxes
            masks_np = masks.cpu().numpy()  # (B, 1, H, W)

            for i in range(batch_size):
                # --- Ground Truth ---
                # mask2box expects (H, W)
                gt_boxes = mask2box(masks_np[i, 0])
                all_true_boxes.append(gt_boxes)

                # --- Prediction ---
                img_pred_boxes = []
                img_pred_scores = []

                # Gating Logic:
                # If predicted study is "Negative for Pneumonia" (index 0), predict no boxes.
                if Config.GATING_ENABLED and study_preds[i] == 0:
                    pass  # Leave lists empty
                else:
                    # Threshold mask
                    binary_mask = (mask_preds_np[i, 0] > 0.5).astype(np.uint8)

                    # Extract boxes
                    boxes = mask2box(binary_mask)

                    for box in boxes:
                        x1, y1, x2, y2 = box
                        # Calculate confidence: Mean probability inside the box
                        # Ensure coordinates are within bounds
                        h, w = mask_preds_np[i, 0].shape
                        x1 = max(0, min(x1, w))
                        x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h))
                        y2 = max(0, min(y2, h))

                        if x2 > x1 and y2 > y1:
                            box_prob_map = mask_preds_np[i, 0, y1:y2, x1:x2]
                            score = np.mean(box_prob_map)

                            img_pred_boxes.append([x1, y1, x2, y2])
                            img_pred_scores.append(float(score))

                all_pred_boxes.append(img_pred_boxes)
                all_pred_scores.append(img_pred_scores)

    avg_loss = total_loss_meter / count

    # Calculate mAP
    map_score = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_true_boxes,
        iou_threshold=Config.IOU_THRESHOLD,
    )

    print(f"Val Loss: {avg_loss:.5f} | Val mAP: {map_score:.10f}")

    return avg_loss, map_score
