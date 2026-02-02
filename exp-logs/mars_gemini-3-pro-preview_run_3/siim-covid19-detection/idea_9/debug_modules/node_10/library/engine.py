import torch
import time
import sys
import numpy as np
from typing import Dict, List, Tuple
from library.config import Config
from library.loss import Criterion
from library.utils import MAPCalculator


class Engine:
    """
    Handles training and evaluation logic for the SwinDyHead model.
    """

    def __init__(self, model, optimizer, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = Criterion().to(device)
        # Initialize GradScaler for Mixed Precision Training
        self.scaler = torch.cuda.amp.GradScaler()

        # Early Stopping parameters
        self.best_map = 0.0
        self.patience = 5
        self.counter = 0
        self.early_stop = False

    def decode_boxes(self, anchors, box_preds):
        """
        Decodes bounding box predictions from ATSS offsets to xyxy coordinates.
        anchors: [N, 4] (cx, cy, stride, stride)
        box_preds: [N, 4] (l, t, r, b)
        Returns: [N, 4] (x1, y1, x2, y2)
        """
        cx, cy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        l, t, r, b = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2], box_preds[:, 3]

        x1 = cx - l * stride
        y1 = cy - t * stride
        x2 = cx + r * stride
        y2 = cy + b * stride

        return torch.stack([x1, y1, x2, y2], dim=1)

    def train_one_epoch(self, dataloader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        box_loss_meter = AverageMeter()
        study_loss_meter = AverageMeter()

        start_time = time.time()

        for step, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)

            # Move targets to device
            # targets is a list of dicts, tensors inside need to be moved
            targets = [
                {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict["loss"]

            # Scaled Backward Pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            cls_loss_meter.update(loss_dict["loss_cls"].item(), batch_size)
            box_loss_meter.update(loss_dict["loss_box"].item(), batch_size)
            study_loss_meter.update(loss_dict["loss_study"].item(), batch_size)

        # Log results
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] Train Loss: {loss_meter.avg:.6f} "
            f"(Cls: {cls_loss_meter.avg:.6f}, Box: {box_loss_meter.avg:.6f}, Study: {study_loss_meter.avg:.6f}) "
            f"Time: {elapsed:.1f}s"
        )

        return loss_meter.avg

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Calculates mAP for detection and Accuracy for study classification.
        """
        self.model.eval()

        map_calculator = MAPCalculator()
        study_correct = 0
        study_total = 0

        start_time = time.time()

        with torch.no_grad():
            for step, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = [
                    {
                        k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                # Forward pass (can use autocast for inference too)
                with torch.cuda.amp.autocast():
                    preds = self.model(images)

                # --- Study Classification Evaluation ---
                study_logits = preds["study_logits"]  # [B, 4]
                study_preds = torch.argmax(study_logits, dim=1)

                for i in range(len(targets)):
                    gt_study = targets[i]["study_label"]
                    if study_preds[i] == gt_study:
                        study_correct += 1
                    study_total += 1

                # --- Detection Evaluation ---
                cls_logits = preds["cls_logits"]  # [B, N, 1]
                bbox_preds = preds["bbox_preds"]  # [B, N, 4]
                anchors = preds["anchors"]  # [N, 4]

                batch_size = cls_logits.size(0)

                for i in range(batch_size):
                    # 1. Get scores
                    scores = cls_logits[i].sigmoid().squeeze(-1)  # [N]

                    # 2. Filter by confidence threshold to speed up mAP calc
                    mask = scores > Config.CONF_THRESHOLD
                    if not mask.any():
                        continue

                    valid_scores = scores[mask]
                    valid_bbox_preds = bbox_preds[i][mask]
                    valid_anchors = anchors[mask]

                    # 3. Decode boxes
                    decoded_boxes = self.decode_boxes(valid_anchors, valid_bbox_preds)

                    # 4. Clip boxes to image size
                    height, width = images.shape[2], images.shape[3]
                    decoded_boxes[:, 0] = decoded_boxes[:, 0].clamp(0, width)
                    decoded_boxes[:, 1] = decoded_boxes[:, 1].clamp(0, height)
                    decoded_boxes[:, 2] = decoded_boxes[:, 2].clamp(0, width)
                    decoded_boxes[:, 3] = decoded_boxes[:, 3].clamp(0, height)

                    # 5. Update mAP calculator
                    gt_boxes = targets[i]["boxes"]

                    # Move to CPU for MAPCalculator (which now uses vectorized ops)
                    pred_boxes_np = decoded_boxes.float().cpu().numpy()
                    pred_scores_np = valid_scores.float().cpu().numpy()
                    gt_boxes_np = gt_boxes.float().cpu().numpy()

                    map_calculator.update(
                        pred_boxes_np, pred_scores_np, gt_boxes_np, iou_threshold=0.5
                    )

        # Compute Metrics
        val_map = map_calculator.compute()
        val_study_acc = study_correct / study_total if study_total > 0 else 0.0

        elapsed = time.time() - start_time
        print(
            f"Validation Results - mAP@0.5: {val_map:.10f}, Study Acc: {val_study_acc:.10f}, Time: {elapsed:.1f}s"
        )

        return val_map, val_study_acc

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_one_epoch(train_loader, epoch)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            # Validate
            val_map, val_acc = self.evaluate(val_loader)

            # Early Stopping Check
            if val_map > self.best_map:
                self.best_map = val_map
                self.counter = 0
                # Save best model
                save_path = f"{Config.WORKING_DIR}/best_model.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"New best mAP! Model saved to {save_path}")
            else:
                self.counter += 1
                print(
                    f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
                break

        print(f"Training complete. Best mAP: {self.best_map:.10f}")


class AverageMeter:
    """Computes and stores the average and current value"""

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
