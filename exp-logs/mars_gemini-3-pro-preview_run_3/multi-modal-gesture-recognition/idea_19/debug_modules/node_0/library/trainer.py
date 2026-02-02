import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import GestureDataset
from library.model import ASK_RN
from library.loss import MultiTaskRefinementLoss
from library.data_utils import seed_everything


class Trainer:
    def __init__(self, load_cached_data: bool = True):
        """
        Initializes the Trainer with model, optimizer, loss, and data loaders.
        Args:
            load_cached_data (bool): Whether to load dataset from cache.
        """
        # Ensure reproducibility
        seed_everything(Config.SEED)

        self.device = Config.DEVICE

        # Initialize Model
        self.model = ASK_RN().to(self.device)

        # Initialize Loss
        self.criterion = MultiTaskRefinementLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Config.LEARNING_RATE
        )

        # Initialize Datasets
        # library.dataset.GestureDataset handles caching logic via library.data_utils
        self.train_dataset = GestureDataset(
            split_name="train", augment=True, load_cached=load_cached_data
        )
        self.val_dataset = GestureDataset(
            split_name="val", augment=False, load_cached=load_cached_data
        )

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Training State
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.model_save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            features = batch["features"].to(self.device)
            cls_labels = batch["cls_labels"].to(self.device)
            bnd_labels = batch["bnd_labels"].to(self.device)

            # Construct targets dictionary
            targets = {"cls_labels": cls_labels, "bnd_labels": bnd_labels}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # Compute loss
            loss, _ = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch} [Train] Loss: {avg_loss}")
        return avg_loss

    def validate(self, epoch):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                cls_labels = batch["cls_labels"].to(self.device)
                bnd_labels = batch["bnd_labels"].to(self.device)

                targets = {"cls_labels": cls_labels, "bnd_labels": bnd_labels}

                outputs = self.model(features)
                loss, _ = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Extract predictions from Stage 3 (Final Refinement)
                # logits_s3: (Batch, Frames, Classes)
                logits = outputs["logits_s3"]
                preds = torch.argmax(logits, dim=2)  # (Batch, Frames)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(cls_labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Flatten for metric calculation
        all_preds_flat = np.concatenate(all_preds).flatten()
        all_targets_flat = np.concatenate(all_targets).flatten()

        # Calculate Accuracy
        accuracy = (all_preds_flat == all_targets_flat).mean()

        # Calculate mIoU
        unique_classes = np.unique(np.concatenate([all_preds_flat, all_targets_flat]))
        iou_list = []

        for c in unique_classes:
            intersection = np.logical_and(
                all_preds_flat == c, all_targets_flat == c
            ).sum()
            union = np.logical_or(all_preds_flat == c, all_targets_flat == c).sum()

            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            iou_list.append(iou)

        mean_iou = np.mean(iou_list) if iou_list else 0.0

        # Print metrics with full precision
        print(
            f"Epoch {epoch} [Val] Loss: {avg_loss} | Accuracy: {accuracy} | mIoU: {mean_iou}"
        )

        return avg_loss, accuracy, mean_iou

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            _ = self.train_epoch(epoch)
            val_loss, _, _ = self.validate(epoch)

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"  New best model saved to {self.model_save_path}")
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
