import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import compute_mcc


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation.
    Loss(x, y) = -alpha_t * (1 - p_t)**gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)

        # Alpha weighting: alpha for class 1, (1-alpha) for class 0
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    Manages the training, validation, and early stopping lifecycle of the model.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.best_mcc = -1.0
        self.best_model_state = None

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in self.train_loader:
            # Move data to device
            x_kin = batch["x_kin"].to(self.device)
            x_vis = batch["x_vis"].to(self.device)
            x_gate = batch["x_gate"].to(self.device)
            x_pos = batch["x_pos"].to(self.device)
            x_team = batch["x_team"].to(self.device)
            y = batch["y"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(x_kin, x_pos, x_team, x_vis, x_gate)
            loss = self.criterion(logits, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                x_kin = batch["x_kin"].to(self.device)
                x_vis = batch["x_vis"].to(self.device)
                x_gate = batch["x_gate"].to(self.device)
                x_pos = batch["x_pos"].to(self.device)
                x_team = batch["x_team"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                logits = self.model(x_kin, x_pos, x_team, x_vis, x_gate)
                loss = self.criterion(logits, y)

                running_loss += loss.item() * y.size(0)
                count += y.size(0)

                # Store probabilities and targets for MCC calculation
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        val_loss = running_loss / count if count > 0 else 0.0

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Calculate MCC using a default threshold of 0.5 for monitoring
        # (Threshold optimization happens after training)
        preds_binary = (all_preds > 0.5).astype(int)
        val_mcc = compute_mcc(all_targets, preds_binary)

        return val_loss, val_mcc, all_targets, all_preds

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_mcc, _, _ = self.validate()

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val MCC: {val_mcc}"
            )

            # Early Stopping Check (Monitor MCC)
            if val_mcc > self.best_mcc:
                self.best_mcc = val_mcc
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                # Save best model to disk immediately
                torch.save(
                    self.best_model_state,
                    os.path.join(Config.WORKING_DIR, "best_model.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best Val MCC: {self.best_mcc}"
                    )
                    break

        # Load best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights.")


def optimize_threshold(y_true, y_pred_probs):
    """
    Performs a grid search to find the classification threshold that maximizes MCC.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_thresh = 0.5
    best_score = -1.0

    # Vectorized search is possible, but loop is clear and memory efficient for large arrays
    for thresh in thresholds:
        y_pred_bin = (y_pred_probs > thresh).astype(int)
        score = compute_mcc(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(
        f"Threshold Optimization - Best Threshold: {best_thresh}, Best MCC: {best_score}"
    )
    return best_thresh


def predict(model, loader, device):
    """
    Generates probability predictions for a dataset.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x_kin = batch["x_kin"].to(device)
            x_vis = batch["x_vis"].to(device)
            x_gate = batch["x_gate"].to(device)
            x_pos = batch["x_pos"].to(device)
            x_team = batch["x_team"].to(device)

            logits = model(x_kin, x_pos, x_team, x_vis, x_gate)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs).flatten()
