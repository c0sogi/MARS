import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from library.config import (
    WORKING_DIR,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    SEED,
)
from library.utils import compute_mcc, seed_everything


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class

        # Alpha balancing
        # If target=1, alpha_t = alpha. If target=0, alpha_t = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term
        focal_term = (1 - pt) ** self.gamma

        loss = alpha_t * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class Trainer:
    def __init__(self, model, device=None):
        seed_everything(SEED)
        self.model = model
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)

        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
        self.best_threshold_path = os.path.join(WORKING_DIR, "best_threshold.npy")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch_idx, (X_kin, X_vis, y) in enumerate(train_loader):
            X_kin = X_kin.to(self.device)
            X_vis = X_vis.to(self.device)
            y = y.to(self.device).unsqueeze(1)  # Ensure shape (Batch, 1)

            self.optimizer.zero_grad()

            logits = self.model(X_kin, X_vis)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for X_kin, X_vis, y in val_loader:
                X_kin = X_kin.to(self.device)
                X_vis = X_vis.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                logits = self.model(X_kin, X_vis)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * y.size(0)

                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        all_logits = np.concatenate(all_logits)
        all_targets = np.concatenate(all_targets)

        return avg_loss, all_logits, all_targets

    def optimize_threshold(self, logits, targets):
        """
        Grid search for the best threshold maximizing MCC.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range 0.01 to 0.99
        thresholds = np.arange(0.01, 1.00, 0.01)

        for thresh in thresholds:
            mcc = compute_mcc(targets, logits, threshold=thresh)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def train(self, train_loader, val_loader, epochs=EPOCHS):
        print(f"Starting training on device: {self.device}")
        best_val_mcc = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_logits, val_targets = self.evaluate(val_loader)

            # Optimize threshold on validation set
            current_best_thresh, current_val_mcc = self.optimize_threshold(
                val_logits, val_targets
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {current_val_mcc:.6f} (Thresh: {current_best_thresh:.2f})"
            )

            # Early Stopping and Model Saving
            if current_val_mcc > best_val_mcc:
                best_val_mcc = current_val_mcc
                patience_counter = 0

                # Save Model
                torch.save(self.model.state_dict(), self.best_model_path)

                # Save Threshold
                np.save(self.best_threshold_path, np.array([current_best_thresh]))
                # print(f"  -> Model saved to {self.best_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val MCC: {best_val_mcc:.6f}")

    def predict(self, test_loader):
        """
        Loads the best model and threshold, runs inference on test_loader.
        Returns binary predictions and raw logits.
        """
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"Best model not found at {self.best_model_path}")

        # Load Model
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Load Threshold
        if os.path.exists(self.best_threshold_path):
            best_thresh = float(np.load(self.best_threshold_path)[0])
        else:
            best_thresh = 0.5
            print("Warning: Best threshold file not found. Defaulting to 0.5.")

        all_logits = []

        with torch.no_grad():
            for X_kin, X_vis, _ in test_loader:
                X_kin = X_kin.to(self.device)
                X_vis = X_vis.to(self.device)

                logits = self.model(X_kin, X_vis)
                all_logits.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits).ravel()

        # Convert logits to probabilities
        probs = np.where(
            all_logits >= 0,
            1.0 / (1.0 + np.exp(-all_logits)),
            np.exp(all_logits) / (1.0 + np.exp(all_logits)),
        )

        predictions = (probs >= best_thresh).astype(int)

        return predictions, all_logits
