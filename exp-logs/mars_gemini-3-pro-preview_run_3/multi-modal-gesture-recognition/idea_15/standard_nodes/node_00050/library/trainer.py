import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

from library.config import Config
from library.model import RSKARN
from library.utils import set_seed


class SmoothingLoss(nn.Module):
    """
    Truncated Mean Squared Error (MSE) loss applied to log-probabilities
    of adjacent frames to enforce temporal smoothness.
    """

    def __init__(self, threshold: float = 1.0):
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (N, L, C) tensor of log probabilities.
        Returns:
            Scalar loss.
        """
        # Calculate difference between t and t-1
        # diff: (N, L-1, C)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Calculate squared error
        sq_error = diff**2

        # Truncate (clamp) the error to prevent exploding gradients from large jumps
        # This makes it robust to genuine transitions
        truncated_error = torch.clamp(sq_error, max=self.threshold)

        return torch.mean(truncated_error)


class CascadedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy for all stages and Smoothing Loss
    for refinement stages.
    """

    def __init__(self, class_weights: list, smoothing_lambda: float):
        super(CascadedLoss, self).__init__()

        # Convert list to tensor for CrossEntropyLoss
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        if torch.cuda.is_available():
            weight_tensor = weight_tensor.cuda()

        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        self.smooth_loss = SmoothingLoss(threshold=1.0)
        self.smoothing_lambda = smoothing_lambda

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary containing logits from model stages.
            targets: (N, L) tensor of ground truth labels.
        Returns:
            Dictionary with total loss and individual components.
        """
        # Flatten targets for CE loss: (N*L)
        targets_flat = targets.view(-1)

        # --- Stage 1 ---
        # Logits: (N, L, C) -> Flatten to (N*L, C)
        logits_1 = outputs["logits_1"]
        loss_1 = self.ce_loss(logits_1.reshape(-1, logits_1.size(-1)), targets_flat)

        # --- Stage 2 ---
        logits_2 = outputs["logits_2"]
        loss_2_ce = self.ce_loss(logits_2.reshape(-1, logits_2.size(-1)), targets_flat)

        # Smoothing on Log Probs of Stage 2
        log_probs_2 = F.log_softmax(logits_2, dim=2)
        loss_2_smooth = self.smooth_loss(log_probs_2)

        loss_2 = loss_2_ce + (self.smoothing_lambda * loss_2_smooth)

        # --- Stage 3 ---
        logits_3 = outputs["logits_3"]
        loss_3_ce = self.ce_loss(logits_3.reshape(-1, logits_3.size(-1)), targets_flat)

        # Smoothing on Log Probs of Stage 3
        log_probs_3 = F.log_softmax(logits_3, dim=2)
        loss_3_smooth = self.smooth_loss(log_probs_3)

        loss_3 = loss_3_ce + (self.smoothing_lambda * loss_3_smooth)

        # Total Loss
        total_loss = loss_1 + loss_2 + loss_3

        return {
            "total": total_loss,
            "loss_1": loss_1,
            "loss_2": loss_2,
            "loss_3": loss_3,
        }


class Trainer:
    """
    Manages the training, validation, and saving of the RSK-ARN model.
    """

    def __init__(
        self, model: RSKARN, train_loader, val_loader, learning_rate: float = 1e-3
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer: Standard Adam as per idea description
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss Function
        self.criterion = CascadedLoss(
            class_weights=Config.CLASS_WEIGHTS, smoothing_lambda=Config.SMOOTHING_LAMBDA
        )

        # Checkpoint path
        self.checkpoint_dir = Config.WORKING_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target, _) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(data)
            loss_dict = self.criterion(outputs, target)
            loss = loss_dict["total"]

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_frames = 0

        with torch.no_grad():
            for data, target, _ in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss_dict = self.criterion(outputs, target)
                total_loss += loss_dict["total"].item()

                # Calculate Accuracy based on Stage 3 probabilities
                probs = outputs["probs_3"]
                preds = torch.argmax(probs, dim=2)  # (N, L)

                correct += (preds == target).sum().item()
                total_frames += target.numel()

        avg_loss = (
            total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        )
        accuracy = correct / total_frames if total_frames > 0 else 0.0

        return avg_loss, accuracy

    def fit(self, epochs: int = Config.NUM_EPOCHS, patience: int = Config.PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs with patience {patience}...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved to {self.best_model_path}")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")
