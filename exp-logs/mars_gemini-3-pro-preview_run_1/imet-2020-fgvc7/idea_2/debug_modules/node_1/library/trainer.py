import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
)
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkModel


class Trainer:
    """
    Manages training, validation, and inference for the Artwork Attribute Labeling task.
    """

    def __init__(self):
        self.device = Config.device
        self.num_classes = Config.num_classes
        self.debug = Config.debug

        # Initialize Model
        print(f"Initializing model: {Config.model_name}...")
        self.model = ArtworkModel(pretrained=Config.pretrained)
        self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.epochs, eta_min=Config.min_lr
        )

        # Initialize Loss Function
        # Construct pos_weight tensor for class imbalance
        pos_weight_tensor = torch.full(
            (self.num_classes,), Config.pos_weight, device=self.device
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Best metrics tracking
        self.best_score = 0.0
        self.best_threshold = 0.5

    def get_dataloader(self, mode):
        """
        Creates a DataLoader for the specified mode.
        """
        transform = get_transforms(data_split=mode)

        if mode == "train":
            csv_path = Config.train_metadata_path
            shuffle = True
        elif mode == "val":
            csv_path = Config.val_metadata_path
            shuffle = False
        else:
            csv_path = Config.test_metadata_path
            shuffle = False

        dataset = ArtworkDataset(
            csv_path=csv_path,
            mode=mode,
            transform=transform,
            load_cached_data=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.batch_size,
            shuffle=shuffle,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=(mode == "train"),
        )

        return dataloader

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply Label Smoothing to targets
            # y_smooth = y * (1 - alpha) + 0.5 * alpha
            if Config.label_smoothing > 0:
                targets_smooth = (
                    targets * (1.0 - Config.label_smoothing)
                    + 0.5 * Config.label_smoothing
                )
            else:
                targets_smooth = targets

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, targets_smooth)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, val_loader):
        """
        Runs validation and optimizes the decision threshold.
        """
        self.model.eval()
        losses = AverageMeter()

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                losses.update(loss.item(), images.size(0))

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate probabilities once
        all_probs = torch.sigmoid(all_logits)

        # Convert targets to numpy for sklearn
        targets_np = all_targets.numpy()

        # Grid Search for Best Threshold
        best_f1 = 0.0
        best_thresh = 0.5

        # Search range: 0.01 to 0.99
        thresholds = np.arange(0.01, 1.00, 0.01)

        for t in thresholds:
            preds = (all_probs > t).float().numpy()
            score = f1_score(targets_np, preds, average="micro", zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_thresh = t

        return losses.avg, best_f1, best_thresh

    def fit(self):
        """
        Main training loop.
        """
        seed_everything(Config.seed)
        Config.setup()

        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        print(f"Starting training for {Config.epochs} epochs...")

        for epoch in range(1, Config.epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_f1, val_thresh = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step()

            # Print Metrics
            print(
                f"Epoch {epoch}/{Config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val F1: {val_f1:.8f} | "
                f"Best Thresh: {val_thresh:.2f}"
            )

            # Save Best Model
            if val_f1 > self.best_score:
                self.best_score = val_f1
                self.best_threshold = val_thresh
                print(f"New best score! Saving model to {Config.model_save_path}")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_f1,
                    Config.model_save_path,
                )

        print(
            f"Training complete. Best Validation F1: {self.best_score:.8f} at Threshold: {self.best_threshold:.2f}"
        )

    def predict(self):
        """
        Generates predictions for the test set using the best model and threshold.
        """
        print("Starting inference on test set...")

        # Load Best Model
        if os.path.exists(Config.model_save_path):
            print(f"Loading best model from {Config.model_save_path}")
            load_checkpoint(self.model, Config.model_save_path, device=self.device)
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()
        test_loader = self.get_dataloader("test")

        results = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device, non_blocking=True)

                # Forward
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Apply Best Threshold
                preds = (probs > self.best_threshold).float().cpu().numpy()

                # Convert binary predictions to attribute_ids strings
                for i in range(len(ids)):
                    img_id = ids[i]
                    pred_row = preds[i]

                    # Get indices where prediction is 1
                    active_indices = np.where(pred_row == 1)[0]

                    # Join into space-separated string
                    attr_str = " ".join(map(str, active_indices))

                    results.append({"id": img_id, "attribute_ids": attr_str})

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Save
        print(f"Saving submission to {Config.submission_path}")
        submission_df.to_csv(Config.submission_path, index=False)
        print("Inference complete.")
