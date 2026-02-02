import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    load_metadata,
    calc_f1_score,
    Logger,
    EarlyStopping,
)
from library.dataset import KuzushijiDataset
from library.model import CenterNetConvNeXt
from library.loss import CenterNetLoss


class Trainer:
    """
    Trainer class to handle the training and validation loop for the Kuzushiji recognition task.
    """

    def __init__(self, debug=Config.DEBUG):
        self.device = Config.DEVICE
        self.debug = debug

        # 1. Load Metadata
        # Using the utility function which handles caching if applicable
        self.train_df = load_metadata(Config.TRAIN_METADATA_PATH)
        self.val_df = load_metadata(Config.VAL_METADATA_PATH)

        if self.debug:
            self.train_df = self.train_df.head(Config.DEBUG_SAMPLE_SIZE)
            self.val_df = self.val_df.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Initialize Datasets
        # Dataset handles parsing and caching of annotations internally
        self.train_dataset = KuzushijiDataset(self.train_df, mode="train")
        self.val_dataset = KuzushijiDataset(self.val_df, mode="val")

        # Dynamically update number of classes based on the dataset
        Config.NUM_CLASSES = self.train_dataset.num_classes

        # 3. Initialize DataLoaders
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

        # 4. Initialize Model
        self.model = CenterNetConvNeXt(num_classes=Config.NUM_CLASSES).to(self.device)

        # 5. Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # 6. Loss Function
        self.criterion = CenterNetLoss()

        # 7. Logging and Early Stopping
        self.logger = Logger()
        self.early_stopping = EarlyStopping(
            patience=5, mode="max", save_path="best_model.pth"
        )

        # Inverse mapping for decoding
        self.idx_to_char = {v: k for k, v in self.train_dataset.char_to_idx.items()}

    def _nms(self, heatmap, kernel=3):
        """
        Performs Non-Maximum Suppression on the heatmap using max pooling.
        """
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heatmap, (kernel, kernel), stride=1, padding=pad
        )
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def _gather_feat(self, feat, ind):
        """
        Gathers features from a feature map at specific indices.
        """
        dim = feat.size(1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.view(feat.size(0), dim, -1).permute(0, 2, 1)
        feat = feat.gather(1, ind)
        return feat

    def _decode(self, hm, wh, reg, cls_logits, K=1200):
        """
        Decodes the CenterNet outputs into bounding box centers and classes.
        """
        batch_size, _, height, width = hm.shape

        # Heatmap -> Sigmoid -> NMS
        hm = torch.sigmoid(hm)
        hm = self._nms(hm)

        # Find top K peaks
        hm = hm.view(batch_size, -1)
        scores, inds = torch.topk(hm, K)

        # Convert indices to grid coordinates
        ys = inds.div(width, rounding_mode="floor").float()
        xs = (inds % width).float()

        # Gather regression offsets
        reg = self._gather_feat(reg, inds)  # (B, K, 2)

        # Apply offsets to grid coordinates
        xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]

        # Gather classification predictions
        cls_feat = self._gather_feat(cls_logits, inds)  # (B, K, C)
        clses = torch.argmax(cls_feat, dim=2).view(batch_size, K, 1)

        # Scale back to original image size (stride 4)
        xs = xs * 4
        ys = ys * 4

        scores = scores.view(batch_size, K, 1)

        # Concatenate results: [x, y, score, class]
        detections = torch.cat([xs, ys, scores, clses.float()], dim=2)

        return detections

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for batch in self.train_loader:
            # Move image to device; targets are moved inside Loss wrapper or handled there
            img = batch["image"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(img)
            loss, _ = self.criterion(outputs, batch)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / n_batches

    def validate(self):
        """
        Runs validation on the validation set and calculates the F1 Score.
        """
        self.model.eval()
        preds_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                img = batch["image"].to(self.device)
                img_ids = batch["image_id"]

                outputs = self.model(img)

                detections = self._decode(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    outputs["cls"],
                    K=Config.MAX_DETECTIONS,
                )

                detections = detections.cpu().numpy()

                # Format predictions for F1 calculation
                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    det = detections[i]

                    # Filter by confidence threshold
                    mask = det[:, 2] >= Config.CONF_THRESHOLD
                    det = det[mask]

                    label_strs = []
                    for d in det:
                        x, y, score, cls_idx = d
                        cls_idx = int(cls_idx)

                        if cls_idx in self.idx_to_char:
                            char = self.idx_to_char[cls_idx]
                            # Format: Unicode X Y
                            label_strs.append(f"{char} {int(x)} {int(y)}")

                    label_str = " ".join(label_strs)
                    preds_list.append({"image_id": img_id, "labels": label_str})

        # Create DataFrame from predictions
        pred_df = pd.DataFrame(preds_list)

        # Calculate F1 Score using the utility function
        # self.val_df contains the ground truth
        f1 = calc_f1_score(self.val_df, pred_df)

        return f1

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_f1 = self.validate()

            self.scheduler.step()

            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.logger.log(metrics)

            # Check early stopping
            self.early_stopping(val_f1, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print(
            f"Training finished. Best Validation F1: {self.early_stopping.best_score}"
        )
