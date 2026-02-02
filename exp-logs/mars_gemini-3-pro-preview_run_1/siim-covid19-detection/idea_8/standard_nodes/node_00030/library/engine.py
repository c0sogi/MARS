import os
import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import seed_everything, mask2bbox, get_map_score


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Handles one training epoch.
    """
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    dataset_size = 0

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    for batch_idx, data in enumerate(loader):
        images = data["image"].to(device)
        masks = data["mask"].to(device)
        labels = data["label"].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits_cls, logits_seg = model(images)

        # Calculate losses
        loss_cls = criterion_cls(logits_cls, labels)
        loss_seg = criterion_seg(logits_seg, masks)

        # Composite loss based on strategy weights
        loss = (Config.CLS_LOSS_WEIGHT * loss_cls) + (Config.SEG_LOSS_WEIGHT * loss_seg)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * batch_size
        running_cls_loss += loss_cls.item() * batch_size
        running_seg_loss += loss_seg.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_cls_loss = running_cls_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size

    return epoch_loss, epoch_cls_loss, epoch_seg_loss


def valid_one_epoch(model, loader, device):
    """
    Handles one validation epoch. Returns losses and predictions for metric calculation.
    """
    model.eval()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    dataset_size = 0

    # storage for metric calculation
    preds_cls_probs = []
    preds_seg_probs = []
    targets_cls = []
    targets_masks = []

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            masks = data["mask"].to(device)
            labels = data["label"].to(device)

            batch_size = images.size(0)

            logits_cls, logits_seg = model(images)

            loss_cls = criterion_cls(logits_cls, labels)
            loss_seg = criterion_seg(logits_seg, masks)

            loss = (Config.CLS_LOSS_WEIGHT * loss_cls) + (
                Config.SEG_LOSS_WEIGHT * loss_seg
            )

            running_loss += loss.item() * batch_size
            running_cls_loss += loss_cls.item() * batch_size
            running_seg_loss += loss_seg.item() * batch_size
            dataset_size += batch_size

            # Store probabilities and targets for mAP/Accuracy calculation
            # Softmax for class, Sigmoid for segmentation
            preds_cls_probs.append(torch.softmax(logits_cls, dim=1).cpu())
            preds_seg_probs.append(torch.sigmoid(logits_seg).cpu())
            targets_cls.append(labels.cpu())
            targets_masks.append(masks.cpu())

    epoch_loss = running_loss / dataset_size
    epoch_cls_loss = running_cls_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size

    # Concatenate all batches
    preds_cls_probs = torch.cat(preds_cls_probs, dim=0)
    preds_seg_probs = torch.cat(preds_seg_probs, dim=0)
    targets_cls = torch.cat(targets_cls, dim=0)
    targets_masks = torch.cat(targets_masks, dim=0)

    return (
        epoch_loss,
        epoch_cls_loss,
        epoch_seg_loss,
        preds_cls_probs,
        preds_seg_probs,
        targets_cls,
        targets_masks,
    )


def fit(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Main training loop with Early Stopping and Metric-based Checkpointing.
    """
    seed_everything(Config.SEED)

    best_score = -float("inf")
    patience = 5
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()

        # --- Training ---
        train_loss, train_cls_loss, train_seg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # --- Validation ---
        (
            val_loss,
            val_cls_loss,
            val_seg_loss,
            val_probs_cls,
            val_probs_seg,
            val_targets_cls,
            val_targets_masks,
        ) = valid_one_epoch(model, val_loader, device)

        # --- Metric Calculation ---
        # 1. Study Accuracy
        val_preds_cls = val_probs_cls.argmax(dim=1)
        val_acc = (val_preds_cls == val_targets_cls).float().mean().item()

        # 2. Image mAP (Opacity)
        # Convert masks to boxes for mAP calculation
        pred_boxes_list = []
        pred_scores_list = []
        true_boxes_list = []

        # Iterate over validation set to extract boxes
        # Note: val_probs_seg is (N, 1, H, W)
        for i in range(len(val_probs_seg)):
            # Coupled Inference Logic (Cite solution_lesson_node_00026)
            # If predicted class is 'Negative for Pneumonia' (index 0), suppress boxes.
            pred_class = val_preds_cls[i].item()

            if pred_class == 0:
                p_boxes = []
                p_scores = []
            else:
                # Prediction
                mask_pred = val_probs_seg[i][0].numpy()
                p_boxes = mask2bbox(mask_pred, threshold=0.5)

                # Calculate score for each box (mean probability inside the box)
                p_scores = []
                for box in p_boxes:
                    x1, y1, x2, y2 = box
                    # Ensure indices are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(mask_pred.shape[1], x2), min(mask_pred.shape[0], y2)

                    if x2 > x1 and y2 > y1:
                        box_slice = mask_pred[y1:y2, x1:x2]
                        score = np.mean(box_slice)
                        p_scores.append(score)
                    else:
                        p_scores.append(0.0)

            # Ground Truth
            mask_gt = val_targets_masks[i][0].numpy()
            t_boxes = mask2bbox(mask_gt, threshold=0.5)

            pred_boxes_list.append(p_boxes)
            pred_scores_list.append(p_scores)
            true_boxes_list.append(t_boxes)

        val_map = get_map_score(
            pred_boxes_list, pred_scores_list, true_boxes_list, iou_threshold=0.5
        )

        # --- Composite Score ---
        # We weight them equally for selection purposes
        composite_score = (val_acc + val_map) / 2.0

        # --- Scheduler Step ---
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(
            f"Train Loss: {train_loss} (Cls: {train_cls_loss}, Seg: {train_seg_loss})"
        )
        print(f"Val Loss: {val_loss} (Cls: {val_cls_loss}, Seg: {val_seg_loss})")
        print(f"Val Acc: {val_acc}")
        print(f"Val mAP: {val_map}")
        print(f"Composite Score: {composite_score}")

        # --- Checkpointing & Early Stopping ---
        if composite_score > best_score:
            best_score = composite_score
            patience_counter = 0
            print(f"New best score! Saving model to {Config.BEST_MODEL_PATH}")
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Composite Score: {best_score}")
