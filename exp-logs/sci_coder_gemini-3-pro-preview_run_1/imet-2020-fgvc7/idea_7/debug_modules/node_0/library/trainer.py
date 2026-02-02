import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import seed_everything, ModelEMA, Mixup, optimize_threshold
from library.data_loader import get_loaders
from library.model import ArtworkModel


class Trainer:
    """
    Trainer class for the Artwork Attribute Labeling task.
    Manages training, validation, and inference using ConvNeXt-Small, Mixup/CutMix, and Model EMA.
    """

    def __init__(self):
        # Set reproducibility
        seed_everything(Config.SEED)

        self.device = Config.DEVICE
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Load Data
        print("Initializing Data Loaders...")
        self.train_loader, self.val_loader, self.test_loader = get_loaders()

        # Initialize Model
        print(f"Initializing Model: {Config.MODEL_NAME}...")
        self.model = ArtworkModel().to(self.device)

        # Initialize EMA (Exponential Moving Average) Model
        # This shadow model will be updated during training and used for validation/inference
        self.ema = ModelEMA(self.model, decay=Config.EMA_DECAY, device=self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Loss Function
        # Using BCEWithLogitsLoss with positive weighting for class imbalance
        pos_weight = torch.tensor([Config.POS_WEIGHT], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Augmentation
        self.mixup = Mixup(
            mixup_alpha=Config.MIXUP_ALPHA,
            cutmix_alpha=Config.CUTMIX_ALPHA,
            prob=Config.MIXUP_PROB,
        )

        # Training State
        self.best_score = 0.0
        self.best_threshold = 0.5
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for i, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Apply Label Smoothing manually before Mixup
            # Target becomes: y * (1 - alpha) + 0.5 * alpha
            labels = (
                labels * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

            # Apply Mixup / CutMix
            images, labels = self.mixup(images, labels)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer Step
            self.optimizer.step()

            # Update EMA Model
            self.ema.update(self.model)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, epoch):
        """
        Evaluates the model on the validation set using the EMA weights.
        """
        # Use EMA model for validation
        eval_model = self.ema.module
        eval_model.eval()

        running_loss = 0.0
        dataset_size = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                outputs = eval_model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

                all_targets.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        # Concatenate results
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Find optimal threshold
        best_thr, best_f1 = optimize_threshold(all_targets, all_preds)

        return val_loss, best_f1, best_thr

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_f1, val_thr = self.validate(epoch)

            # Step Scheduler
            self.scheduler.step()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val F1: {val_f1:.9f} | "
                f"Best Thr: {val_thr:.2f}"
            )

            # Save Best Model
            if val_f1 > self.best_score:
                self.best_score = val_f1
                self.best_threshold = val_thr
                print(f"New best F1 score! Saving model to {self.best_model_path}")

                torch.save(
                    {
                        "model_state_dict": self.ema.module.state_dict(),
                        "threshold": self.best_threshold,
                        "score": self.best_score,
                        "epoch": epoch,
                    },
                    self.best_model_path,
                )

        print(
            f"Training finished. Best F1: {self.best_score:.9f} at Threshold: {self.best_threshold:.2f}"
        )

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        """
        print("Starting inference on test set...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            # Load weights into the base model instance
            self.model.load_state_dict(checkpoint["model_state_dict"])
            threshold = checkpoint["threshold"]
            print(
                f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')} with threshold {threshold:.2f}"
            )
        else:
            print(
                "Warning: No best model checkpoint found. Using current EMA model state."
            )
            # Copy EMA weights to base model for inference
            self.model.load_state_dict(self.ema.module.state_dict())
            threshold = 0.5

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, _, ids in self.test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                # Binarize predictions using the optimal threshold
                preds = (probs > threshold).int().cpu().numpy()

                for i, img_id in enumerate(ids):
                    # Get indices of active attributes
                    active_indices = np.where(preds[i] == 1)[0]

                    # Format as space-separated string
                    pred_str = " ".join(map(str, active_indices))
                    results.append({"id": img_id, "attribute_ids": pred_str})

        # Save Submission
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path} ({len(df_sub)} rows).")
