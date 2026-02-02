import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
from library.config import Config
from library.utils import calculate_map


class Engine:
    """
    Encapsulates training and validation loops for the ResNet34-FPN model.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss Functions
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_seg = nn.BCEWithLogitsLoss()

        self.best_map = 0.0

        # Mapping for study labels
        self.study_class_map = {
            0: "negative",
            1: "typical",
            2: "indeterminate",
            3: "atypical",
        }

    def train_one_epoch(self, data_loader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        scaler = torch.cuda.amp.GradScaler()

        for batch_idx, (images, masks, labels) in enumerate(data_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            # Convert one-hot labels to class indices for CrossEntropy
            target_cls = torch.argmax(labels, dim=1)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                cls_logits, seg_logits = self.model(images)

                loss_cls = self.criterion_cls(cls_logits, target_cls)
                loss_seg = self.criterion_seg(seg_logits, masks)

                # Weighted Composite Loss
                loss = (Config.LOSS_WEIGHT_CLASS * loss_cls) + (
                    Config.LOSS_WEIGHT_SEG * loss_seg
                )

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def validate(self, data_loader):
        """
        Evaluates the model on the validation set and calculates mAP.
        """
        self.model.eval()
        total_loss = 0

        pred_rows = []
        gt_rows = []
        idx_counter = 0

        with torch.no_grad():
            for images, masks, labels in data_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                target_cls = torch.argmax(labels, dim=1)

                # Forward Pass
                cls_logits, seg_logits = self.model(images)

                # Loss Calculation
                loss_cls = self.criterion_cls(cls_logits, target_cls)
                loss_seg = self.criterion_seg(seg_logits, masks)
                loss = (Config.LOSS_WEIGHT_CLASS * loss_cls) + (
                    Config.LOSS_WEIGHT_SEG * loss_seg
                )
                total_loss += loss.item()

                # --- Prepare Data for mAP Calculation ---

                # Study Predictions (Softmax)
                cls_probs = torch.softmax(cls_logits, dim=1)
                pred_cls_inds = torch.argmax(cls_probs, dim=1).cpu().numpy()
                pred_cls_confs = torch.max(cls_probs, dim=1).values.cpu().numpy()

                # Segmentation Predictions (Sigmoid)
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()  # (B, 1, H, W)

                # Ground Truth
                gt_cls_inds = target_cls.cpu().numpy()
                gt_masks = masks.cpu().numpy()  # (B, 1, H, W)

                batch_size = images.size(0)

                for i in range(batch_size):
                    # Generate temporary IDs for metric calculation
                    # (Metric only cares about matching IDs between GT and Pred)
                    img_id = f"{idx_counter}_image"
                    study_id = f"{idx_counter}_study"
                    idx_counter += 1

                    # 1. Study Level
                    # GT
                    gt_cls_name = self.study_class_map[gt_cls_inds[i]]
                    gt_rows.append(
                        {"id": study_id, "PredictionString": f"{gt_cls_name} 1 0 0 1 1"}
                    )

                    # Pred
                    pred_cls_name = self.study_class_map[pred_cls_inds[i]]
                    pred_conf = pred_cls_confs[i]
                    pred_rows.append(
                        {
                            "id": study_id,
                            "PredictionString": f"{pred_cls_name} {pred_conf} 0 0 1 1",
                        }
                    )

                    # 2. Image Level
                    # GT (From Mask)
                    gt_mask = gt_masks[i, 0]
                    gt_str = self._mask_to_string(gt_mask, threshold=0.5, conf=1.0)
                    gt_rows.append({"id": img_id, "PredictionString": gt_str})

                    # Pred (Gated Logic)
                    if pred_cls_name == "negative":
                        # Gating: If study is negative, force no opacity
                        pred_str = Config.NONE_PREDICTION
                    else:
                        pred_mask = seg_probs[i, 0]
                        pred_str = self._mask_to_string(
                            pred_mask, threshold=Config.SEG_THRESHOLD, conf=None
                        )

                    pred_rows.append({"id": img_id, "PredictionString": pred_str})

        avg_loss = total_loss / len(data_loader)

        # Calculate mAP
        pred_df = pd.DataFrame(pred_rows)
        gt_df = pd.DataFrame(gt_rows)
        map_score = calculate_map(pred_df, gt_df)

        return avg_loss, map_score

    def _mask_to_string(self, mask, threshold=0.5, conf=None):
        """
        Converts a probability mask to a prediction string using contours.
        Args:
            mask: (H, W) float array
            threshold: Threshold for binarization
            conf: If provided, used as confidence. If None, mean pixel value is used.
        """
        mask_binary = (mask > threshold).astype(np.uint8)

        if mask_binary.sum() == 0:
            return Config.NONE_PREDICTION

        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        res = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter extremely small noise
            if w * h < 10:
                continue

            x1, y1, x2, y2 = x, y, x + w, y + h

            if conf is None:
                # Use mean probability in the box as confidence
                box_mask = mask[y1:y2, x1:x2]
                c = np.mean(box_mask) if box_mask.size > 0 else 0.0
            else:
                c = conf

            res.append(f"opacity {c:.4f} {x1} {y1} {x2} {y2}")

        if not res:
            return Config.NONE_PREDICTION

        return " ".join(res)


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, num_epochs, device
):
    """
    Main driver function for training.
    """
    engine = Engine(model, device, optimizer, scheduler)

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = engine.train_one_epoch(train_loader)
        val_loss, val_map = engine.validate(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f} Val mAP: {val_map:.6f}"
        )

        # Metric-based Checkpointing
        if val_map > engine.best_map:
            engine.best_map = val_map
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best mAP! Model saved to {Config.CHECKPOINT_PATH}")
