import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.loss import FocalLoss
from library.utils import optimize_threshold


class Trainer:
    """
    Manages the training lifecycle of the KCVR-Net model, including
    training loops, validation, metric calculation, and early stopping.
    """

    def __init__(self, model, device=None):
        """
        Args:
            model (torch.nn.Module): The KCVRNet model to train.
            device (torch.device, optional): The device to train on. Defaults to Config.DEVICE.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = model.to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: Focal Loss
        self.criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

        # State for Early Stopping
        self.best_mcc = -1.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, max_batches=None):
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): The training data loader.
            max_batches (int, optional): Limit number of batches for debugging.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for i, (X_kin, X_vis, y) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break

            X_kin = X_kin.to(self.device)
            X_vis = X_vis.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(X_kin, X_vis)
            loss = self.criterion(logits, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader, max_batches=None):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): The validation data loader.
            max_batches (int, optional): Limit number of batches for debugging.

        Returns:
            tuple: (average_loss, best_mcc, best_threshold)
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (X_kin, X_vis, y) in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break

                X_kin = X_kin.to(self.device)
                X_vis = X_vis.to(self.device)
                y = y.to(self.device)

                logits = self.model(X_kin, X_vis)
                loss = self.criterion(logits, y)

                running_loss += loss.item() * y.size(0)
                count += y.size(0)

                # Store probabilities and targets for MCC optimization
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            best_thresh, best_mcc = optimize_threshold(all_targets, all_preds)
        else:
            best_thresh, best_mcc = 0.5, 0.0

        return avg_loss, best_mcc, best_thresh

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, max_batches=None):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            max_batches (int, optional): Limit batches per epoch for debugging.
        """
        print(f"Starting training on device: {self.device}")

        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, max_batches=max_batches)
            val_loss, val_mcc, val_thresh = self.validate(
                val_loader, max_batches=max_batches
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MCC: {val_mcc} | "
                f"Best Thresh: {val_thresh}"
            )

            # Early Stopping Logic
            if val_mcc > self.best_mcc:
                print(
                    f"Validation MCC improved from {self.best_mcc} to {val_mcc}. Saving model..."
                )
                self.best_mcc = val_mcc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement in MCC. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MCC: {self.best_mcc}")
