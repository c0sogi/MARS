import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from library.utils import get_logger, save_checkpoint, load_checkpoint
from library.loss import FocalLoss


class Trainer:
    """
    Trainer class for training, validating, and predicting with the Swin Transformer model.
    """

    def __init__(
        self,
        cfg,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler=None,
        label_map=None,
    ):
        """
        Args:
            cfg: Configuration object.
            model: PyTorch model.
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            test_loader: DataLoader for testing.
            optimizer: PyTorch optimizer.
            scheduler: Learning rate scheduler (optional).
            label_map: Dictionary mapping index to raw category_id (inverse map).
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_map = label_map
        self.device = cfg.device
        self.logger = get_logger(cfg.log_path)

        # Loss function
        self.criterion = FocalLoss(gamma=2.0)

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Mixup / CutMix parameters
        self.mixup_alpha = 0.4
        self.mixup_prob = 0.5

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Apply Mixup Augmentation manually to work with FocalLoss (hard targets)
            # We mix inputs and then compute loss as linear combination of losses for both targets
            use_mixup = False
            if np.random.rand() < self.mixup_prob:
                use_mixup = True
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                index = torch.randperm(images.size(0)).to(self.device)

                mixed_images = lam * images + (1 - lam) * images[index]
                target_a, target_b = labels, labels[index]

            with autocast():
                if use_mixup:
                    outputs = self.model(mixed_images)
                    # Mixup loss: weighted sum of focal loss for both targets
                    loss = lam * self.criterion(outputs, target_a) + (
                        1 - lam
                    ) * self.criterion(outputs, target_b)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            float: Macro F1 score.
        """
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs = self.model(images)

                # Get predictions
                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        # Calculate Macro F1
        macro_f1 = f1_score(targets, preds, average="macro")
        return macro_f1

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_f1 = -1.0
        patience_counter = 0

        self.logger.info("Starting training...")

        for epoch in range(self.cfg.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_f1 = self.validate()

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # Log metrics (Full precision for F1)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.cfg.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Macro F1: {val_f1} - "
                f"Time: {elapsed:.0f}s"
            )

            # Checkpoint & Early Stopping
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_f1": best_f1,
                    },
                    is_best,
                    self.cfg.working_dir,
                )
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        self.logger.info(f"Training complete. Best Val F1: {best_f1}")

    def predict(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        Loads the best model weights before predicting.
        """
        self.logger.info("Loading best model for inference...")

        # Load best model
        try:
            load_checkpoint(self.cfg.best_model_path, self.model, device=self.device)
        except FileNotFoundError:
            self.logger.warning("Best model not found. Using current model weights.")

        self.model.eval()
        ids = []
        predictions = []

        self.logger.info("Generating predictions...")

        with torch.no_grad():
            for images, image_ids in self.test_loader:
                images = images.to(self.device)

                with autocast():
                    outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                ids.extend(image_ids.cpu().numpy())

                preds_cpu = predicted.cpu().numpy()
                if self.label_map:
                    preds_mapped = [self.label_map[p] for p in preds_cpu]
                    predictions.extend(preds_mapped)
                else:
                    predictions.extend(preds_cpu)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Save to CSV
        submission_df.to_csv(self.cfg.submission_path, index=False)
        self.logger.info(f"Submission saved to {self.cfg.submission_path}")
