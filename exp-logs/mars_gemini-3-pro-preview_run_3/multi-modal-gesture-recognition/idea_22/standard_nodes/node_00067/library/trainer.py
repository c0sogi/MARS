import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import get_device, set_seeds


class Trainer:
    """
    Trainer class for the Residual Log-Kinematic Refinement Network (RLK-RN).
    Manages training, validation, loss computation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Args:
            model (nn.Module): The RLK-RN model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        """
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer: Adam (not AdamW)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: NLLLoss (since model outputs LogSoftmax)
        # Class weights: 0.2 for background, 1.0 for others
        class_weights = Config.get_class_weights_tensor(self.device)
        self.criterion = nn.NLLLoss(weight=class_weights, reduction="mean")

        # Hyperparameters
        self.smoothing_lambda = Config.SMOOTHING_LAMBDA
        self.smoothing_threshold_sq = Config.SMOOTHING_THRESHOLD**2
        self.loss_weights = Config.LOSS_WEIGHTS

    def compute_loss(self, outputs, targets):
        """
        Computes the Cascaded Loss + Truncated MSE Smoothing Loss.

        Args:
            outputs (list): [l1, l2, l3] log-probabilities from the model.
            targets (torch.Tensor): Ground truth labels (Batch, Time).

        Returns:
            torch.Tensor: Total combined loss.
            dict: Dictionary of individual loss components for logging.
        """
        l1, l2, l3 = outputs

        # Flatten targets for NLLLoss: (Batch * Time)
        # Reshape outputs: (Batch * Time, NumClasses)
        B, T, C = l1.shape
        targets_flat = targets.view(-1)

        loss_l1 = self.criterion(l1.view(-1, C), targets_flat)
        loss_l2 = self.criterion(l2.view(-1, C), targets_flat)
        loss_l3 = self.criterion(l3.view(-1, C), targets_flat)

        # --- Truncated MSE Smoothing Loss ---
        # Applied to Refinement Stages (l2, l3)
        # Calculate diff between t and t-1

        def smoothing_loss(log_probs):
            # log_probs: (Batch, Time, Classes)
            # diff: (Batch, Time-1, Classes)
            diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]
            sq_diff = diff**2
            # Truncate
            truncated_sq_diff = torch.clamp(sq_diff, max=self.smoothing_threshold_sq)
            return torch.mean(truncated_sq_diff)

        smooth_l2 = smoothing_loss(l2)
        smooth_l3 = smoothing_loss(l3)

        # --- Total Loss ---
        # Weighted sum of classification losses
        total_cls_loss = (
            self.loss_weights["stage1"] * loss_l1
            + self.loss_weights["stage2"] * loss_l2
            + self.loss_weights["stage3"] * loss_l3
        )

        # Weighted sum of smoothing losses
        total_smooth_loss = self.smoothing_lambda * (smooth_l2 + smooth_l3)

        total_loss = total_cls_loss + total_smooth_loss

        loss_dict = {
            "loss_l1": loss_l1.item(),
            "loss_l2": loss_l2.item(),
            "loss_l3": loss_l3.item(),
            "smooth_l2": smooth_l2.item(),
            "smooth_l3": smooth_l3.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        # Track individual components for monitoring
        running_components = {
            "loss_l1": 0.0,
            "loss_l2": 0.0,
            "loss_l3": 0.0,
            "smooth_l2": 0.0,
            "smooth_l3": 0.0,
        }

        start_time = time.time()
        batches = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)  # [l1, l2, l3]

            # Compute loss
            loss, loss_dict = self.compute_loss(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            running_loss += loss.item()
            for k, v in loss_dict.items():
                if k != "total":
                    running_components[k] += v

            batches += 1

        avg_loss = running_loss / batches if batches > 0 else 0.0
        duration = time.time() - start_time

        # Print summary
        print(f"Epoch {epoch_idx} [Train] Loss: {avg_loss:.6f} | Time: {duration:.2f}s")
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and frame accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct_frames = 0
        total_frames = 0
        batches = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)  # [l1, l2, l3]

                # Use Stage 3 output for metrics
                final_log_probs = outputs[2]

                # Compute loss
                loss, _ = self.compute_loss(outputs, labels)
                running_loss += loss.item()

                # Compute Frame Accuracy
                preds = torch.argmax(final_log_probs, dim=2)  # (Batch, Time)

                correct = (preds == labels).sum().item()
                total = labels.numel()

                correct_frames += correct
                total_frames += total
                batches += 1

        avg_loss = running_loss / batches if batches > 0 else 0.0
        accuracy = correct_frames / total_frames if total_frames > 0 else 0.0

        print(f"Epoch Val   [Valid] Loss: {avg_loss:.6f} | Frame Acc: {accuracy:.6f}")
        return avg_loss, accuracy

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        # Ensure model directory exists
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            _ = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved ({best_val_loss:.6f} -> {val_loss:.6f}). Saving model..."
                )
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")
