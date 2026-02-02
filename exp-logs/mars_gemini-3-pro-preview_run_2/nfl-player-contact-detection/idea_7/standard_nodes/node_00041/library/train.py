import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import FocalLoss, compute_mcc, optimize_threshold
from library.data_processing import load_and_process_data
from library.model import KinematicMLP


class Trainer:
    """
    Manages the training, validation, and optimization lifecycle of the Kinematic MLP model.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Model
        self.model = KinematicMLP(config).to(self.device)

        # Initialize Optimizer (AdamW)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        # Initialize Loss Function (Focal Loss)
        self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    def train_epoch(self, train_loader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns:
            avg_loss: Average Focal Loss
            mcc: Matthews Correlation Coefficient (using 0.5 threshold)
            y_true: Ground truth labels
            y_prob: Predicted probabilities
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        y_prob = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        # Calculate MCC using a default threshold of 0.5 for monitoring progress
        y_pred_bin = (y_prob >= 0.5).astype(int)
        mcc = compute_mcc(y_true, y_pred_bin)

        return avg_loss, mcc, y_true, y_prob

    def fit(self, train_dataset, val_dataset):
        """
        Executes the training loop with Early Stopping and Threshold Optimization.
        """
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        best_mcc = -1.0
        patience_counter = 0
        best_model_path = os.path.join(self.config.artifact_dir, "best_model.pth")

        print(f"Starting training on device: {self.device}")
        print(
            f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Training Step
            train_loss = self.train_epoch(train_loader)

            # Validation Step
            val_loss, val_mcc, _, _ = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MCC: {val_mcc} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping Logic
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  New best model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.config.patience}"
                )

            if patience_counter >= self.config.patience:
                print("Early stopping triggered.")
                break

        # Post-Training: Load best model and optimize threshold
        print("Loading best model for threshold optimization...")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        # Get predictions from the best model
        _, final_mcc, y_true, y_prob = self.validate(val_loader)
        print(f"Final Validation MCC (Default Threshold): {final_mcc}")

        # Optimize Threshold
        best_threshold = optimize_threshold(y_true, y_prob)

        # Save the optimal threshold
        threshold_path = os.path.join(self.config.artifact_dir, "best_threshold.npy")
        np.save(threshold_path, np.array([best_threshold]))
        print(f"Best threshold {best_threshold} saved to {threshold_path}")

        return best_threshold


def train_model(debug=False):
    """
    Orchestrates the entire training pipeline:
    1. Config setup and seeding
    2. Data loading and processing
    3. Model training and validation
    4. Artifact saving
    """
    # 1. Configuration and Seeding
    config = Config()
    config.debug = debug
    set_seed(config.seed)

    # 2. Load Data
    print("Loading training data...")
    train_dataset, _ = load_and_process_data(split="train", debug=debug)

    print("Loading validation data...")
    val_dataset, _ = load_and_process_data(split="validation", debug=debug)

    # 3. Initialize Trainer and Fit
    trainer = Trainer(config)
    best_threshold = trainer.fit(train_dataset, val_dataset)

    return best_threshold
