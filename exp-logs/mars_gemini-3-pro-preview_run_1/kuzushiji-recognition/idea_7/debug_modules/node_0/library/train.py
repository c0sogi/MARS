import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.dataset import KuzushijiDataset
from library.model import ConvNeXtCenterNet
from library.loss import CenterNetLoss
from library.utils import (
    Tracker,
    save_checkpoint,
    decode_predictions,
    calc_f1_score,
    LabelEncoder,
)


class Trainer:
    def __init__(self):
        # 1. Setup Device and Seed
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # 2. Data Loaders
        print("Initializing Datasets...")
        self.train_dataset = KuzushijiDataset(mode="train", load_cached_data=True)
        self.val_dataset = KuzushijiDataset(mode="val", load_cached_data=True)

        # Handle Debug Mode
        if Config.DEBUG:
            print(
                f"Debug Mode: Limiting datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            indices = np.arange(Config.DEBUG_SAMPLE_SIZE)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Access Label Encoder from dataset (needed for decoding)
        # If Subset is used, access the underlying dataset
        if isinstance(self.val_dataset, torch.utils.data.Subset):
            self.le = self.val_dataset.dataset.le
        else:
            self.le = self.val_dataset.le

        # 3. Model
        print(f"Initializing Model: {Config.MODEL_NAME}")
        self.model = ConvNeXtCenterNet(num_classes=Config.NUM_CLASSES, pretrained=True)
        self.model.to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # 5. Loss
        self.criterion = CenterNetLoss()

        # 6. Tracking
        self.best_f1 = 0.0
        self.log_path = os.path.join(Config.WORK_DIR, "training_log.csv")
        self.log_header = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_precision",
            "val_recall",
            "val_f1",
            "lr",
        ]

        # Initialize log file
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=self.log_header).to_csv(self.log_path, index=False)

    def train_epoch(self, epoch):
        self.model.train()
        losses = Tracker()
        loss_hm_tracker = Tracker()
        loss_reg_tracker = Tracker()
        loss_cls_tracker = Tracker()

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            # Forward
            outputs = self.model(batch["image"])

            # Loss
            loss, loss_stats = self.criterion(outputs, batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update Trackers
            losses.update(loss.item(), batch["image"].size(0))
            loss_hm_tracker.update(loss_stats["loss_hm"], batch["image"].size(0))
            loss_reg_tracker.update(loss_stats["loss_reg"], batch["image"].size(0))
            loss_cls_tracker.update(loss_stats["loss_cls"], batch["image"].size(0))

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train] "
            f"Loss: {losses.avg:.6f} (HM: {loss_hm_tracker.avg:.4f}, "
            f"Reg: {loss_reg_tracker.avg:.4f}, Cls: {loss_cls_tracker.avg:.4f}) "
            f"Time: {elapsed:.0f}s"
        )

        return losses.avg

    def validate(self, epoch):
        self.model.eval()
        losses = Tracker()

        # For F1 Calculation
        all_pred_strs = []
        all_gt_strs = []

        start_time = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward
                outputs = self.model(batch["image"])

                # Validation Loss
                loss, _ = self.criterion(outputs, batch)
                losses.update(loss.item(), batch["image"].size(0))

                # Decode Predictions for Metric Calculation
                # outputs: hm, reg_wh, cls_logits
                hm = outputs["hm"]
                hm = torch.sigmoid(hm)

                # Split reg_wh into offset and wh
                # reg_wh is (B, 4, H, W) -> [off_x, off_y, w, h]
                reg = outputs["reg_wh"][:, 0:2, :, :]
                wh = outputs["reg_wh"][:, 2:4, :, :]

                # Decode
                # dets shape: (batch, K, 6) -> [x, y, w, h, score, class_id]
                # x, y are in feature map coordinates
                dets = decode_predictions(hm, reg, wh, K=Config.MAX_DETECTIONS)

                # Process batch for metric strings
                batch_size = dets.size(0)

                for i in range(batch_size):
                    # Ground Truth String
                    gt_str = batch["label_str"][i]
                    all_gt_strs.append(gt_str)

                    # Prediction String Construction
                    pred_parts = []

                    # Filter by confidence
                    valid_mask = dets[i, :, 4] >= Config.CONF_THRESHOLD
                    valid_dets = dets[i][valid_mask]

                    # If too many, take top K (though decode already does top K, we filter by conf)
                    if valid_dets.size(0) > Config.MAX_DETECTIONS:
                        _, top_indices = torch.topk(
                            valid_dets[:, 4], Config.MAX_DETECTIONS
                        )
                        valid_dets = valid_dets[top_indices]

                    for det in valid_dets:
                        # det: [x, y, w, h, score, cls_id]
                        # Scale coordinates back to original image size
                        # Down ratio is 4
                        x = det[0].item() * 4
                        y = det[1].item() * 4

                        cls_idx = int(det[5].item())
                        char_label = self.le.inverse_transform(cls_idx)

                        if char_label != "":
                            # Format: Label X Y
                            pred_parts.append(f"{char_label} {int(x)} {int(y)}")

                    all_pred_strs.append(" ".join(pred_parts))

        # Calculate Metrics
        metrics = calc_f1_score(all_pred_strs, all_gt_strs)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]   "
            f"Loss: {losses.avg:.6f} | "
            f"Precision: {metrics['precision']:.10f} | "
            f"Recall: {metrics['recall']:.10f} | "
            f"F1: {metrics['f1']:.10f} | "
            f"Time: {elapsed:.0f}s"
        )

        return losses.avg, metrics

    def fit(self):
        print(f"Starting training on {self.device}...")
        patience = 5
        no_improve_epochs = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
            }
            pd.DataFrame([log_entry]).to_csv(
                self.log_path, mode="a", header=False, index=False
            )

            # Checkpoint & Early Stopping
            if val_metrics["f1"] > self.best_f1:
                print(
                    f"Validation F1 improved from {self.best_f1:.6f} to {val_metrics['f1']:.6f}. Saving model..."
                )
                self.best_f1 = val_metrics["f1"]
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_f1": self.best_f1,
                    },
                    is_best=True,
                    filepath=Config.BEST_MODEL_PATH,
                )
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"No improvement in F1. Patience: {no_improve_epochs}/{patience}")

            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best F1: {self.best_f1:.10f}")
