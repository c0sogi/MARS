import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import ChestXrayDataset, get_transforms
from library.model import AntiAliasedResNetUNet
from library.utils import (
    seed_everything,
    AverageMeter,
    ModelEMA,
    calculate_map,
    calculate_classification_ap,
)


def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes.
    """
    images = []
    masks = []
    study_labels = []
    boxes = []
    box_labels = []
    study_ids = []
    image_ids = []

    for item in batch:
        images.append(item["image"])
        masks.append(item["mask"])
        study_labels.append(item["study_label"])
        boxes.append(item["boxes"])
        box_labels.append(item["box_labels"])
        study_ids.append(item["study_id"])
        image_ids.append(item["image_id"])

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    study_labels = torch.stack(study_labels, dim=0)

    return {
        "image": images,
        "mask": masks,
        "study_label": study_labels,
        "boxes": boxes,  # List of tensors
        "box_labels": box_labels,  # List of tensors
        "study_id": study_ids,
        "image_id": image_ids,
    }


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        # 1. Prepare Data
        print("Initializing Datasets...")
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)

        if Config.DEBUG:
            print("DEBUG Mode: Using small subset of data.")
            df_train = df_train.head(Config.BATCH_SIZE * 2)
            df_val = df_val.head(Config.BATCH_SIZE * 2)

        self.train_ds = ChestXrayDataset(
            df_train, mode="train", transform=get_transforms("train")
        )
        self.val_ds = ChestXrayDataset(
            df_val, mode="val", transform=get_transforms("val")
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # 2. Prepare Model
        print("Initializing Model...")
        self.model = AntiAliasedResNetUNet().to(self.device)

        # EMA Setup
        self.ema = None
        if Config.USE_EMA:
            print(f"Enabling EMA with decay {Config.EMA_DECAY}")
            self.ema = ModelEMA(self.model, decay=Config.EMA_DECAY, device=self.device)

        # 3. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
        )

        # 4. Losses
        # Study: Multi-class classification (Mutually Exclusive)
        self.criterion_cls = nn.CrossEntropyLoss()
        # Image: Pixel-wise binary classification
        self.criterion_seg = nn.BCEWithLogitsLoss()

        self.best_score = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        cls_losses = AverageMeter()
        seg_losses = AverageMeter()

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            # labels are (B, 4) one-hot-like float. Convert to indices for CrossEntropy
            study_labels = batch["study_label"].to(self.device)
            study_targets = torch.argmax(study_labels, dim=1)

            self.optimizer.zero_grad()

            cls_logits, seg_logits = self.model(images)

            # Calculate Losses
            loss_cls = self.criterion_cls(cls_logits, study_targets)
            loss_seg = self.criterion_seg(seg_logits, masks)

            total_loss = (Config.LOSS_WEIGHT_CLS * loss_cls) + (
                Config.LOSS_WEIGHT_SEG * loss_seg
            )

            total_loss.backward()
            self.optimizer.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

            # Logging
            batch_size = images.size(0)
            losses.update(total_loss.item(), batch_size)
            cls_losses.update(loss_cls.item(), batch_size)
            seg_losses.update(loss_seg.item(), batch_size)

        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Train Loss: {losses.avg:.4f} "
            f"(Cls: {cls_losses.avg:.4f}, Seg: {seg_losses.avg:.4f})"
        )

        return losses.avg

    def validate(self):
        # Use EMA model for validation if available
        eval_model = self.ema.module if self.ema else self.model
        eval_model.eval()

        # Metrics Storage
        # Study Level
        all_study_targets = []
        all_study_probs = []

        # Image Level (Detection)
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                study_labels = batch["study_label"]  # Keep on CPU for metrics
                gt_boxes_batch = batch["boxes"]  # List of tensors
                gt_labels_batch = batch["box_labels"]  # List of tensors

                cls_logits, seg_logits = eval_model(images)

                # --- Study Level Processing ---
                probs = torch.softmax(cls_logits, dim=1).cpu()
                all_study_targets.append(study_labels)
                all_study_probs.append(probs)

                # Determine predicted class for gating
                pred_classes = torch.argmax(probs, dim=1)

                # --- Image Level Processing (Box Extraction) ---
                # Apply sigmoid
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

                batch_size = images.size(0)

                for i in range(batch_size):
                    # Ground Truth
                    all_gt_boxes.append(gt_boxes_batch[i])
                    all_gt_labels.append(gt_labels_batch[i])

                    # Prediction Extraction
                    # Gating: If predicted "Negative for Pneumonia" (Index 0), no boxes.
                    if Config.GATED_PREDICTION and pred_classes[i].item() == 0:
                        all_pred_boxes.append(torch.tensor([], dtype=torch.float32))
                        all_pred_scores.append(torch.tensor([], dtype=torch.float32))
                        all_pred_labels.append(torch.tensor([], dtype=torch.int64))
                        continue

                    # Extract boxes from mask
                    mask = seg_probs[i, 0]  # (H, W)
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    boxes = []
                    scores = []

                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Filter tiny boxes
                        if w * h < 50:
                            continue

                        # Box format: xmin, ymin, xmax, ymax
                        boxes.append([x, y, x + w, y + h])

                        # Score: Mean probability in the box area
                        box_score = np.mean(mask[y : y + h, x : x + w])
                        scores.append(box_score)

                    if len(boxes) > 0:
                        all_pred_boxes.append(torch.tensor(boxes, dtype=torch.float32))
                        all_pred_scores.append(
                            torch.tensor(scores, dtype=torch.float32)
                        )
                        # All boxes are class 0 ('opacity')
                        all_pred_labels.append(
                            torch.zeros(len(boxes), dtype=torch.int64)
                        )
                    else:
                        all_pred_boxes.append(torch.tensor([], dtype=torch.float32))
                        all_pred_scores.append(torch.tensor([], dtype=torch.float32))
                        all_pred_labels.append(torch.tensor([], dtype=torch.int64))

        # --- Calculate Metrics ---

        # 1. Study mAP
        all_study_targets = torch.cat(all_study_targets, dim=0).numpy()
        all_study_probs = torch.cat(all_study_probs, dim=0).numpy()

        study_aps = calculate_classification_ap(all_study_targets, all_study_probs)
        study_map = np.mean(study_aps)

        # 2. Image mAP (IoU > 0.5)
        image_map = calculate_map(
            all_pred_boxes,
            all_pred_scores,
            all_pred_labels,
            all_gt_boxes,
            all_gt_labels,
            num_classes=1,
            iou_threshold=0.5,
        )

        # 3. Composite Score
        composite_score = (study_map + image_map) / 2.0

        print(f"Validation Metrics:")
        print(f"  Study mAP: {study_map:.10f}")
        print(f"  Image mAP: {image_map:.10f}")
        print(f"  Composite: {composite_score:.10f}")

        return composite_score

    def fit(self):
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            self.train_epoch(epoch)

            # Validate
            score = self.validate()

            # Scheduler Step
            self.scheduler.step()

            # Checkpoint
            if score > self.best_score:
                print(
                    f"Score improved ({self.best_score:.6f} -> {score:.6f}). Saving model..."
                )
                self.best_score = score
                # Save state dict
                state = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "ema_state_dict": (
                        self.ema.module.state_dict() if self.ema else None
                    ),
                    "optimizer": self.optimizer.state_dict(),
                    "best_score": self.best_score,
                }
                torch.save(state, Config.MODEL_SAVE_PATH)

            print("-" * 30)

        print(f"Training Complete. Best Composite Score: {self.best_score:.6f}")
