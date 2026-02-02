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
from library.model import ResNet18UNet
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
    Executes validation epoch and computes mAP across all classes.
    """
    model.eval()
    running_loss = 0.0

    # Lists to store full validation set predictions/targets for mAP
    all_preds = []
    all_targets = []

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

            # --- Prepare Data for mAP ---
            probs_cls = torch.softmax(logit_cls, dim=1).cpu().numpy()
            probs_mask = torch.sigmoid(logit_mask).cpu().numpy()

            gt_labels = labels_indices.cpu().numpy()
            gt_masks = masks.cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                # Per-image containers
                p_boxes = []
                p_scores = []
                p_labels = []

                t_boxes = []
                t_labels = []

                # 1. Study Classes (0-3)
                # Treat as 1-pixel box [0, 0, 1, 1]
                for c in range(4):
                    # Prediction
                    p_boxes.append([0, 0, 1, 1])
                    p_scores.append(probs_cls[i, c])
                    p_labels.append(c)

                # GT Study
                gt_c = gt_labels[i]
                t_boxes.append([0, 0, 1, 1])
                t_labels.append(gt_c)

                # 2. Opacity Class (4)
                # Prediction
                pm = probs_mask[i, 0]
                det_boxes = mask2bbox(pm, threshold=0.5)
                for box in det_boxes:
                    x1, y1, x2, y2 = box
                    # Clip
                    x1 = max(0, min(x1, Config.IMG_SIZE - 1))
                    y1 = max(0, min(y1, Config.IMG_SIZE - 1))
                    x2 = max(0, min(x2, Config.IMG_SIZE))
                    y2 = max(0, min(y2, Config.IMG_SIZE))

                    if x2 > x1 and y2 > y1:
                        score = np.mean(pm[y1:y2, x1:x2])
                        p_boxes.append([x1, y1, x2, y2])
                        p_scores.append(score)
                        p_labels.append(4)  # Class 4 is Opacity

                # GT Opacity
                gm = gt_masks[i, 0]
                gt_det_boxes = mask2bbox(gm, threshold=0.5)
                for box in gt_det_boxes:
                    t_boxes.append(box)
                    t_labels.append(4)

                # Convert to numpy and store
                all_preds.append(
                    {
                        "boxes": np.array(p_boxes, dtype=float),
                        "scores": np.array(p_scores, dtype=float),
                        "labels": np.array(p_labels, dtype=int),
                    }
                )
                all_targets.append(
                    {
                        "boxes": np.array(t_boxes, dtype=float),
                        "labels": np.array(t_labels, dtype=int),
                    }
                )

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate mAP across all 5 classes (0-3 Study, 4 Opacity)
    val_map = calculate_map(all_preds, all_targets, num_classes=5, iou_threshold=0.5)

    return epoch_loss, val_map


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
    model = ResNet18UNet(num_classes=Config.NUM_CLASSES, pretrained=True)
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
