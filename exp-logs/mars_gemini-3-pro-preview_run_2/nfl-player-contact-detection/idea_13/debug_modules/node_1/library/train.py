import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.loss import FocalLoss
from library.model import WideResNetMLP
from library.dataset import get_dataloader
from library.feature_engineering import generate_train_val_features


class Trainer:
    """
    Manages the training lifecycle of the Entity-Centric Wide-Residual-MLP.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.optimizer = None
        self.criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA)

        # Early Stopping State
        self.best_mcc = -1.0
        self.patience_counter = 0
        self.best_model_state = None

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for features, targets in train_loader:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass (returns logits)
            logits = self.model(features).squeeze(1)

            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            count += features.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss, true labels, and predicted probabilities.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(features).squeeze(1)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * features.size(0)
                count += features.size(0)

                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        y_true = np.concatenate(all_targets)
        y_probs = np.concatenate(all_probs)

        return avg_loss, y_true, y_probs

    def fit(self, load_cached_data=True):
        """
        Main execution method to load data, initialize model, and run training loop.
        """
        seed_everything(Config.SEED)

        print("--- Starting Training Pipeline ---")

        # 1. Load Data
        X_train, y_train, X_val, y_val = generate_train_val_features(
            load_cached_data=load_cached_data
        )

        print(f"Train set shape: {X_train.shape}")
        print(f"Val set shape: {X_val.shape}")

        # 2. Prepare DataLoaders
        train_loader = get_dataloader(
            X_train,
            y_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = get_dataloader(
            X_val,
            y_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # 3. Initialize Model
        input_dim = X_train.shape[1]
        self.model = WideResNetMLP(input_dim=input_dim).to(self.device)

        # 4. Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Training Loop
        print(f"Training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, y_true, y_probs = self.validate(val_loader)

            # Optimize Threshold for Metric Calculation
            # We use the utility to find the best MCC for this epoch to guide early stopping
            # This ensures we don't stop early just because the default threshold (0.5) is suboptimal
            best_thresh, epoch_mcc = optimize_threshold(y_true, y_probs)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {epoch_mcc} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping Logic
            if epoch_mcc > self.best_mcc:
                self.best_mcc = epoch_mcc
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0

                # Save best model immediately
                torch.save(self.best_model_state, Config.MODEL_PATH)
                print(f"  -> New Best Model Saved! MCC: {self.best_mcc}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print("Training complete.")
        print(f"Best Validation MCC: {self.best_mcc}")


def run_training():
    """
    Entry point function to run the training process.
    """
    trainer = Trainer()
    trainer.fit(load_cached_data=Config.CACHE_DATA)
