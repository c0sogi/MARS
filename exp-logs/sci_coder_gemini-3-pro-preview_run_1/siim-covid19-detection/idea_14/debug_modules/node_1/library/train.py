import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader

from library.config import cfg
from library.dataset import SIIMDataset, process_and_cache_data
from library.model import ResNet18D_UNet
from library.loss import CompositeLoss
from library.utils import AverageMeter, calculate_map, seed_everything


def extract_boxes_from_prob(prob_map, threshold=0.5):
    """
    Extracts bounding boxes and scores from a probability map.
    Args:
        prob_map (np.array): Float array (H, W) with values in [0, 1].
        threshold (float): Threshold for binarization.
    Returns:
        boxes (list): List of [x1, y1, x2, y2].
        scores (list): List of confidence scores.
    """
    mask = (prob_map > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    scores = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter very small artifacts
        if w * h < 10:
            continue

        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        boxes.append([x1, y1, x2, y2])

        # Score is the mean probability within the mask region
        roi = prob_map[y : y + h, x : x + w]
        score = roi.mean()
        scores.append(float(score))

    return boxes, scores


def train_one_epoch(
    loader, model, criterion, optimizer, scheduler, scaler, device, epoch
):
    """
    Handles one epoch of training.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    study_losses = AverageMeter()
    image_losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)

        # Unpack targets
        cls_targets = targets["study_label"].to(device)
        seg_targets = targets["mask"].to(device)

        # Forward pass with Mixed Precision
        with amp.autocast():
            cls_logits, seg_logits = model(images)
            loss, study_loss, image_loss = criterion(
                cls_logits, seg_logits, cls_targets, seg_targets
            )

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        study_losses.update(study_loss.item(), images.size(0))
        image_losses.update(image_loss.item(), images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Step scheduler at epoch level
    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{cfg.epochs}] Train Loss: {losses.avg:.4f} "
        f"(Study: {study_losses.avg:.4f}, Image: {image_losses.avg:.4f}) "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    return losses.avg


def valid_one_epoch(loader, model, criterion, device):
    """
    Handles validation and mAP calculation.
    """
    losses = AverageMeter()
    study_acc = AverageMeter()

    model.eval()

    # Containers for mAP calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_targets = targets["study_label"].to(device)
            seg_targets = targets["mask"].to(device)

            # Forward
            cls_logits, seg_logits = model(images)
            loss, _, _ = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
            losses.update(loss.item(), images.size(0))

            # --- Study Metrics ---
            cls_preds = torch.argmax(cls_logits, dim=1)
            acc = (cls_preds == cls_targets).float().mean()
            study_acc.update(acc.item(), images.size(0))

            # --- Segmentation / Detection Metrics ---
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy()

            # Process batch for mAP
            batch_size = images.size(0)
            for b in range(batch_size):
                # 1. Extract Targets
                # Note: dataset.py returns labels as 1s. calculate_map checks class index 0 if num_classes=1.
                # We map target labels to 0.
                gt_boxes = targets["boxes"][b]  # Tensor (N, 4)
                gt_labels = torch.zeros(
                    (gt_boxes.shape[0],), dtype=torch.int64
                )  # Force class 0

                all_targets.append({"boxes": gt_boxes, "labels": gt_labels})

                # 2. Extract Predictions
                # Gating: If predicted study is "Negative for Pneumonia" (index 0), suppress boxes
                pred_study_cls = cls_preds[b].item()

                if pred_study_cls == 0:
                    # Negative prediction -> No boxes
                    p_boxes = torch.zeros((0, 4), dtype=torch.float32)
                    p_scores = torch.zeros((0,), dtype=torch.float32)
                    p_labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    # Extract boxes from mask
                    raw_boxes, raw_scores = extract_boxes_from_prob(
                        seg_probs[b, 0], threshold=0.5
                    )

                    if len(raw_boxes) > 0:
                        p_boxes = torch.tensor(raw_boxes, dtype=torch.float32)
                        p_scores = torch.tensor(raw_scores, dtype=torch.float32)
                        p_labels = torch.zeros(
                            (len(raw_boxes),), dtype=torch.int64
                        )  # Class 0
                    else:
                        p_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        p_scores = torch.zeros((0,), dtype=torch.float32)
                        p_labels = torch.zeros((0,), dtype=torch.int64)

                all_preds.append(
                    {"boxes": p_boxes, "scores": p_scores, "labels": p_labels}
                )

    # Calculate mAP
    # num_classes=1 checks class index 0. We aligned preds and targets to class 0.
    map_score = calculate_map(
        all_preds, all_targets, iou_threshold=cfg.iou_threshold, num_classes=1
    )

    print(
        f"Validation Loss: {losses.avg:.4f} | Study Acc: {study_acc.avg:.4f} | Detection mAP: {map_score:.10f}"
    )

    return losses.avg, study_acc.avg, map_score


def train():
    """
    Main training pipeline.
    """
    seed_everything(cfg.seed)

    # 1. Data Preparation
    print("Loading and processing data...")
    train_df, train_imgs, train_masks, train_dims = process_and_cache_data(
        "train", load_cached_data=True
    )
    val_df, val_imgs, val_masks, val_dims = process_and_cache_data(
        "val", load_cached_data=True
    )

    train_dataset = SIIMDataset(
        train_df, train_imgs, train_masks, train_dims, split="train"
    )
    val_dataset = SIIMDataset(val_df, val_imgs, val_masks, val_dims, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 2. Model Setup
    print(f"Initializing {cfg.backbone} U-Net...")
    model = ResNet18D_UNet().to(cfg.device)

    criterion = CompositeLoss().to(cfg.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    scaler = amp.GradScaler()

    # 3. Training Loop
    best_score = 0.0

    print(f"Starting training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        # Train
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            cfg.device,
            epoch,
        )

        # Validate
        val_loss, val_acc, val_map = valid_one_epoch(
            val_loader, model, criterion, cfg.device
        )

        # Composite Score: Average of Study Accuracy and Detection mAP
        # This balances the multi-task nature for checkpointing
        composite_score = (val_acc + val_map) / 2.0

        print(f"Epoch {epoch+1} Composite Score: {composite_score:.6f}")

        # Checkpoint
        if composite_score > best_score:
            print(
                f"Score Improved ({best_score:.6f} -> {composite_score:.6f}). Saving model..."
            )
            best_score = composite_score
            torch.save(model.state_dict(), cfg.model_save_path)

    print(f"Training complete. Best Composite Score: {best_score:.6f}")
    print(f"Model saved to {cfg.model_save_path}")
