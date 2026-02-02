import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.loss import FocalLoss


class Trainer:
    """
    Manages the training lifecycle of the TD-SRV-Net model.
    Includes training loop, validation with threshold optimization,
    early stopping, and checkpointing.
    """

    def __init__(self, model, device=None):
        self.config = Config
        seed_everything(self.config.SEED)

        self.device = device if device is not None else self.config.DEVICE
        self.model = model.to(self.device)

        # Initialize Loss Function (Focal Loss for class imbalance)
        self.criterion = FocalLoss(
            alpha=self.config.FOCAL_LOSS_ALPHA,
            gamma=self.config.FOCAL_LOSS_GAMMA,
            reduction="mean",
        )

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )

        # Training State
        self.best_mcc = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0

        # Artifact Paths
        self.model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")
        self.threshold_path = os.path.join(
            self.config.WORKING_DIR, "best_threshold.npy"
        )

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in train_loader:
            # Unpack batch: ((x_kin, x_vis, x_cat), y)
            (x_kin, x_vis, x_cat), targets = batch

            x_kin = x_kin.to(self.device)
            x_vis = x_vis.to(self.device)
            x_cat = x_cat.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns logits)
            logits = self.model(x_kin, x_vis, x_cat)

            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

        return running_loss / count if count > 0 else 0.0

    def evaluate(self, val_loader):
        """Evaluates the model on the validation set and optimizes the threshold."""
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                (x_kin, x_vis, x_cat), targets = batch

                x_kin = x_kin.to(self.device)
                x_vis = x_vis.to(self.device)
                x_cat = x_cat.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(x_kin, x_vis, x_cat)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * targets.size(0)
                count += targets.size(0)

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        # Concatenate all batches
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)

        # Optimize threshold for MCC
        best_thresh, best_mcc = optimize_threshold(all_targets, all_probs)

        return avg_loss, best_mcc, best_thresh

    def train(self, train_loader, val_loader, epochs=None):
        """
        Main training loop with Early Stopping.
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mcc, val_thresh = self.evaluate(val_loader)

            # Print metrics (Full precision for Val MCC as requested)
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MCC: {val_mcc} | Threshold: {val_thresh:.4f}"
            )

            # Early Stopping and Checkpointing
            if val_mcc > self.best_mcc:
                self.best_mcc = val_mcc
                self.best_threshold = val_thresh
                self.patience_counter = 0

                # Save Best Model
                torch.save(self.model.state_dict(), self.model_path)

                # Save Best Threshold
                np.save(self.threshold_path, np.array([self.best_threshold]))

                print(f"New best model saved with MCC: {self.best_mcc}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best MCC: {self.best_mcc}")

    def predict(self, test_loader, load_best_model=True):
        """
        Generates probability predictions for the test set.
        """
        if load_best_model and os.path.exists(self.model_path):
            print(f"Loading best model from {self.model_path} for prediction...")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                (x_kin, x_vis, x_cat), _ = batch

                x_kin = x_kin.to(self.device)
                x_vis = x_vis.to(self.device)
                x_cat = x_cat.to(self.device)

                logits = self.model(x_kin, x_vis, x_cat)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)
