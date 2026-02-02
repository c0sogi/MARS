import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import TrainConfig, ModelConfig
from library.utils import set_seed, mixup_criterion, compute_auc, calculate_pos_weight
from library.dataset import get_dataloaders
from library.model import WhaleDetector


class Trainer:
    """
    Trainer class for Right Whale Call Detection.
    Handles model initialization, training loop, validation, and inference.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the Trainer.

        Args:
            load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
        """
        self.device = TrainConfig.device
        set_seed(TrainConfig.seed)

        print(f"Initializing Trainer on device: {self.device}")

        # 1. Load Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # 2. Calculate Class Weights
        # Extract labels from the dataset to compute inverse frequency weight
        train_labels = self.train_loader.dataset.labels
        self.pos_weight = calculate_pos_weight(train_labels).to(self.device)
        print(f"Calculated Positive Class Weight: {self.pos_weight.item():.4f}")

        # 3. Initialize Model
        self.model = WhaleDetector()
        self.model.to(self.device)

        # 4. Loss Function
        # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # 5. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TrainConfig.lr,
            weight_decay=TrainConfig.weight_decay,
        )

        # 6. Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=TrainConfig.epochs, eta_min=TrainConfig.min_lr
        )

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            # Reshape labels to (B, 1) to match model output
            labels = labels.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            # Mixup Augmentation
            use_mixup = TrainConfig.mixup_alpha > 0
            if use_mixup:
                # Sample lambda from Beta distribution
                lam = np.random.beta(TrainConfig.mixup_alpha, TrainConfig.mixup_alpha)

                # Shuffle indices for mixing
                index = torch.randperm(images.size(0)).to(self.device)

                # Mix images
                mixed_images = lam * images + (1 - lam) * images[index]

                # Get corresponding labels
                y_a, y_b = labels, labels[index]

                # Forward pass
                outputs = self.model(mixed_images)

                # Compute mixed loss
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            else:
                # Standard forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns: (avg_loss, auc_score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

                # Store for metric calculation
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        auc = compute_auc(all_targets, all_preds)

        return avg_loss, auc

    def fit(self, epochs=TrainConfig.epochs, patience=5):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        best_auc = 0.0
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs (Patience: {patience})...")

        for epoch in range(epochs):
            start_time = time.time()

            # 1. Train
            train_loss = self.train_one_epoch(epoch)

            # 2. Validate
            val_loss, val_auc = self.validate()

            # 3. Update Scheduler
            self.scheduler.step()

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # 4. Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save Best Model
                torch.save(self.model.state_dict(), TrainConfig.CHECKPOINT_PATH)
                print(f"  New Best AUC! Model saved to {TrainConfig.CHECKPOINT_PATH}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"\nEarly stopping triggered after {patience} epochs of no improvement."
                )
                break

        print(f"Training complete. Best Validation AUC: {best_auc}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to submission.csv.
        """
        print("\nGenerating predictions on Test set...")

        # 1. Load Best Model Weights
        if not os.path.exists(TrainConfig.CHECKPOINT_PATH):
            print("Warning: No checkpoint found. Using current model weights.")
        else:
            print(f"Loading weights from {TrainConfig.CHECKPOINT_PATH}")
            self.model.load_state_dict(
                torch.load(TrainConfig.CHECKPOINT_PATH, map_location=self.device)
            )

        self.model.eval()
        all_preds = []

        # 2. Inference Loop
        with torch.no_grad():
            for images in self.test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                # Flatten to 1D array
                all_preds.extend(probs.cpu().numpy().flatten())

        # 3. Match with Clip Names
        test_df = pd.read_csv(TrainConfig.TEST_CSV)

        # Handle Debug Slicing to ensure alignment
        if TrainConfig.debug:
            test_df = test_df.head(TrainConfig.debug_samples)

        if len(all_preds) != len(test_df):
            print(
                f"Error: Prediction count ({len(all_preds)}) matches metadata ({len(test_df)}) mismatch."
            )
            # Fallback for safety: truncate to shorter length
            min_len = min(len(all_preds), len(test_df))
            test_df = test_df.iloc[:min_len]
            all_preds = all_preds[:min_len]

        # 4. Create Submission File
        submission = pd.DataFrame(
            {"clip": test_df["clip_name"], "probability": all_preds}
        )

        # 5. Save
        submission.to_csv(TrainConfig.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {TrainConfig.SUBMISSION_PATH}")
        print(submission.head())
