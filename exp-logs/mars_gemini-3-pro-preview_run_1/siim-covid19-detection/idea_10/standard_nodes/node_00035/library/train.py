import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from library.config import Config
from library.dataset import prepare_data, SIIMDataset, get_transforms
from library.model import ResNet18UNetMultiScale
from library.utils import seed_everything, mask2bbox, calculate_map


def train_fn(model, loader, optimizer, criterion_cls, criterion_seg, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks, labels, _) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Convert one-hot labels to indices for CrossEntropyLoss
        labels_indices = torch.argmax(labels, dim=1)

        optimizer.zero_grad()

        logit_mask, logit_cls = model(images)

        # Calculate weighted loss
        loss_cls = criterion_cls(logit_cls, labels_indices)
        loss_seg = criterion_seg(logit_mask, masks)

        loss = (Config.CLS_LOSS_WEIGHT * loss_cls) + (Config.SEG_LOSS_WEIGHT * loss_seg)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def eval_fn(model, loader, criterion_cls, criterion_seg, device):
    """
    Executes validation epoch and computes metrics.
    Returns: loss, study_accuracy, opacity_map
    """
    model.eval()
    running_loss = 0.0

    study_preds = []
    study_targets = []

    # Lists for mAP calculation
    pred_boxes_list = []
    true_boxes_list = []

    with torch.no_grad():
        for batch_idx, (images, masks, labels, _) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logit_mask, logit_cls = model(images)

            labels_indices = torch.argmax(labels, dim=1)

            loss_cls = criterion_cls(logit_cls, labels_indices)
            loss_seg = criterion_seg(logit_mask, masks)
            loss = (Config.CLS_LOSS_WEIGHT * loss_cls) + (
                Config.SEG_LOSS_WEIGHT * loss_seg
            )
            running_loss += loss.item() * images.size(0)

            # --- Metrics Collection ---

            # 1. Study Classification
            probs_cls = torch.softmax(logit_cls, dim=1)
            preds_cls = torch.argmax(probs_cls, dim=1)

            study_preds.extend(preds_cls.cpu().numpy())
            study_targets.extend(labels_indices.cpu().numpy())

            # 2. Segmentation / Object Detection
            probs_mask = torch.sigmoid(logit_mask)
            probs_mask_np = probs_mask.cpu().numpy()
            gt_masks_np = masks.cpu().numpy()

            batch_size = images.size(0)
            preds_cls_np = preds_cls.cpu().numpy()

            for i in range(batch_size):
                # Predicted Boxes
                img_pred_boxes = []

                # Apply Gating Logic: If predicted class is Negative (0), suppress boxes.
                # Cite solution_lesson_node_00026
                if preds_cls_np[i] != 0:
                    pm = probs_mask_np[i, 0]
                    p_boxes = mask2bbox(pm, threshold=0.5)

                    for box in p_boxes:
                        x1, y1, x2, y2 = box
                        # Clip to image bounds
                        x1 = max(0, min(x1, Config.IMG_SIZE - 1))
                        y1 = max(0, min(y1, Config.IMG_SIZE - 1))
                        x2 = max(0, min(x2, Config.IMG_SIZE))
                        y2 = max(0, min(y2, Config.IMG_SIZE))

                        if x2 > x1 and y2 > y1:
                            # Score is the mean probability within the box
                            score = np.mean(pm[y1:y2, x1:x2])
                            img_pred_boxes.append([x1, y1, x2, y2, score])

                pred_boxes_list.append(img_pred_boxes)

                # Ground Truth Boxes
                gm = gt_masks_np[i, 0]
                g_boxes = mask2bbox(gm, threshold=0.5)
                true_boxes_list.append(g_boxes)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate Study Accuracy
    study_preds = np.array(study_preds)
    study_targets = np.array(study_targets)
    study_acc = np.mean(study_preds == study_targets)

    # Calculate Opacity mAP
    opacity_map = calculate_map(pred_boxes_list, true_boxes_list, iou_threshold=0.5)

    return epoch_loss, study_acc, opacity_map


def run_training(debug=False):
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)

    # 1. Data Loading
    print("Preparing data...")
    train_images, train_masks, train_labels, _ = prepare_data(
        "train", load_cached_data=True, debug=debug
    )
    val_images, val_masks, val_labels, _ = prepare_data(
        "val", load_cached_data=True, debug=debug
    )

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = SIIMDataset(
        train_images, train_masks, train_labels, transforms=train_transforms
    )
    val_dataset = SIIMDataset(
        val_images, val_masks, val_labels, transforms=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    device = torch.device(Config.DEVICE)
    model = ResNet18UNetMultiScale(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 4. Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_composite_score = -1.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_fn(
            model, train_loader, optimizer, criterion_cls, criterion_seg, device
        )
        val_loss, val_acc, val_map = eval_fn(
            model, val_loader, criterion_cls, criterion_seg, device
        )

        scheduler.step()

        # Composite Score: Average of Study Accuracy and Opacity mAP
        composite_score = (val_acc + val_map) / 2.0

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.0f}s")
        print(f"  Train Loss: {train_loss:.5f}")
        print(
            f"  Val Loss:   {val_loss:.5f} | Study Acc: {val_acc:.5f} | Opacity mAP: {val_map:.5f}"
        )
        print(f"  Composite Score: {composite_score:.5f}")

        if composite_score > best_composite_score:
            print(
                f"  >>> Improved score from {best_composite_score:.5f} to {composite_score:.5f}. Saving model..."
            )
            best_composite_score = composite_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break

    print(f"Training complete. Best Composite Score: {best_composite_score:.5f}")
