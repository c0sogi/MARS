import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import SwinCenterNet
from library.loss import CenterNetLoss
from library.utils import decode_centernet_predictions, kuzushiji_f1_score


class Trainer:
    def __init__(self, config=Config):
        self.config = config
        self.device = self.config.DEVICE
        self.set_seed(self.config.SEED)

        # Directories
        os.makedirs(self.config.WORK_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

        # Data Loading
        self.train_df = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        self.val_df = pd.read_csv(self.config.VAL_METADATA_PATH)

        # Pre-calculate validation image dimensions for metric calculation
        # This avoids I/O during the validation loop
        self.val_dims = []
        for _, row in self.val_df.iterrows():
            path = os.path.join(self.config.INPUT_DIR, row["file_path"])
            # We only need dims, reading header is enough if possible, but cv2 reads full
            # Given dataset size (~1k val), reading is acceptable once
            img = cv2.imread(path)
            if img is not None:
                self.val_dims.append((img.shape[0], img.shape[1]))  # H, W
            else:
                self.val_dims.append((self.config.IMG_SIZE, self.config.IMG_SIZE))

        # Datasets
        self.train_dataset = KuzushijiDataset(mode="train", load_cached_data=True)
        self.val_dataset = KuzushijiDataset(mode="val", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,  # Must be False to align with metadata/dims
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Model & Training Components
        self.model = SwinCenterNet().to(self.device)
        self.criterion = CenterNetLoss().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.NUM_EPOCHS, eta_min=1e-6
        )

        self.scaler = GradScaler()

        # Class mapping for decoding
        _, self.id2char = self.config.get_class_mapping()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def fit(self):
        best_f1 = 0.0
        patience = 5
        patience_counter = 0
        best_model_path = os.path.join(self.config.WORK_DIR, "best_model.pth")

        print(f"Starting training for {self.config.NUM_EPOCHS} epochs...")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()
            val_f1 = val_metrics["f1"]

            # Scheduler Step
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val F1: {val_f1:.6f} | "
                f"Precision: {val_metrics['precision']:.6f} | "
                f"Recall: {val_metrics['recall']:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with F1: {best_f1:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best F1: {best_f1:.6f}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Move batch to device
            img = batch["image"].to(self.device)
            # Targets are moved inside loss function logic or here
            # The loss function in library.loss expects batch on CPU/GPU but moves them internally?
            # Checking library.loss: it does `batch["hm"].to(device)`.
            # So we just pass the batch dict.

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(img)
                loss, _ = self.criterion(outputs, batch)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / n_batches if n_batches > 0 else 0.0

    def restore_coordinates(self, x, y, orig_h, orig_w):
        """
        Maps coordinates from 1024x1024 model output back to original image space.
        Reverses Albumentations LongestMaxSize + PadIfNeeded.
        """
        # 1. Determine the scaling factor used
        scale = self.config.IMG_SIZE / max(orig_h, orig_w)

        # 2. Determine the padded dimensions
        resized_h = int(round(orig_h * scale))
        resized_w = int(round(orig_w * scale))

        # 3. Determine padding
        pad_top = (self.config.IMG_SIZE - resized_h) // 2
        pad_left = (self.config.IMG_SIZE - resized_w) // 2

        # 4. Reverse padding
        x_unpad = x - pad_left
        y_unpad = y - pad_top

        # 5. Reverse scaling
        x_orig = x_unpad / scale
        y_orig = y_unpad / scale

        return x_orig, y_orig

    def validate(self):
        self.model.eval()

        pred_strs = []
        gt_strs = []

        # We iterate sequentially. The loader is not shuffled.
        # We track the global index to fetch metadata.
        global_idx = 0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device)
                batch_size = imgs.size(0)

                # Forward
                outputs = self.model(imgs)

                # Decode: (B, K, 7) -> [x, y, w, h, score, class_id, raw_hm]
                # x, y are in 1024x1024 pixels
                detections = decode_centernet_predictions(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    outputs["cls_logits"],
                    K=self.config.MAX_DETECTIONS,
                )

                detections = detections.cpu().numpy()

                for i in range(batch_size):
                    # Get original dimensions
                    orig_h, orig_w = self.val_dims[global_idx]

                    # Get GT string
                    gt_str = self.val_df.iloc[global_idx]["labels"]
                    if pd.isna(gt_str):
                        gt_str = ""
                    gt_strs.append(gt_str)

                    # Process predictions
                    det = detections[i]
                    # Filter by confidence
                    det = det[det[:, 4] >= self.config.CONF_THRESHOLD]

                    current_preds = []
                    for d in det:
                        x_pred, y_pred = d[0], d[1]
                        score = d[4]
                        cls_id = int(d[5])

                        # Restore coordinates
                        x_orig, y_orig = self.restore_coordinates(
                            x_pred, y_pred, orig_h, orig_w
                        )

                        # Clip to image bounds
                        x_orig = max(0, min(x_orig, orig_w))
                        y_orig = max(0, min(y_orig, orig_h))

                        label = self.id2char[cls_id]

                        # Format: Label X Y
                        current_preds.append(f"{label} {int(x_orig)} {int(y_orig)}")

                    pred_strs.append(" ".join(current_preds))
                    global_idx += 1

        # Calculate Metric
        metrics = kuzushiji_f1_score(pred_strs, gt_strs)
        return metrics

    def predict(self):
        print("Starting inference on test set...")
        best_model_path = os.path.join(self.config.WORK_DIR, "best_model.pth")

        if not os.path.exists(best_model_path):
            print(
                "No best model found. Using current model state (likely untrained or interrupted)."
            )
        else:
            print(f"Loading model from {best_model_path}")
            state_dict = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        test_dataset = KuzushijiDataset(mode="test", load_cached_data=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(self.device)
                img_ids = batch["image_id"]
                orig_hs = batch["orig_h"].numpy()
                orig_ws = batch["orig_w"].numpy()

                outputs = self.model(imgs)

                detections = decode_centernet_predictions(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    outputs["cls_logits"],
                    K=self.config.MAX_DETECTIONS,
                )
                detections = detections.cpu().numpy()

                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    orig_h = orig_hs[i]
                    orig_w = orig_ws[i]

                    det = detections[i]
                    det = det[det[:, 4] >= self.config.CONF_THRESHOLD]

                    pred_parts = []
                    for d in det:
                        x_pred, y_pred = d[0], d[1]
                        cls_id = int(d[5])

                        x_orig, y_orig = self.restore_coordinates(
                            x_pred, y_pred, orig_h, orig_w
                        )

                        x_orig = max(0, min(x_orig, orig_w))
                        y_orig = max(0, min(y_orig, orig_h))

                        label = self.id2char[cls_id]
                        pred_parts.append(f"{label} {int(x_orig)} {int(y_orig)}")

                    results.append({"image_id": img_id, "labels": " ".join(pred_parts)})

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
