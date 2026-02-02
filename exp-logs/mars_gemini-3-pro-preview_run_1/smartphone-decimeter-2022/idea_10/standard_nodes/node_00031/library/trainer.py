import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from library.config import Config


class Trainer:
    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (str): Device to run training on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.criterion = nn.L1Loss(
            reduction="none"
        )  # MAE Loss, reduction handled manually with mask
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (ReduceLROnPlateau is a good default for convergence)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Early Stopping
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.min_delta = Config.EARLY_STOPPING_MIN_DELTA
        self.best_score = float("inf")
        self.counter = 0
        self.early_stop = False

        # Checkpoint path
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (features, targets, mask, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Model expects (B, C, L), returns (B, Out, L)
            outputs = self.model(features, mask)

            # Rearrange outputs to match targets (B, L, Out)
            outputs = outputs.permute(0, 2, 1)

            # Compute Loss
            # targets shape: (B, L, 2)
            # mask shape: (B, L)
            loss_elementwise = self.criterion(outputs, targets)  # (B, L, 2)

            # Apply mask to ignore padding
            # Expand mask to (B, L, 2)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss_elementwise)

            # Sum loss over valid elements and divide by number of valid elements
            masked_loss = (loss_elementwise * mask_expanded).sum()
            valid_elements = mask_expanded.sum()

            loss = masked_loss / (valid_elements + 1e-8)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_CLIP_NORM
            )

            self.optimizer.step()

            # Statistics
            running_loss += masked_loss.item()
            total_samples += valid_elements.item()

        if total_samples == 0:
            return float("nan")

        avg_loss = running_loss / total_samples
        return avg_loss

    def validate(self):
        """
        Runs validation and calculates the competition metric.
        Metric: Mean of (50th + 95th percentile) distance errors, averaged over phones.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        # Containers for metric calculation
        phone_errors = defaultdict(list)

        with torch.no_grad():
            for features, targets, mask, meta_list in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)
                outputs = outputs.permute(0, 2, 1)  # (B, L, 2)

                # Compute MAE Loss for monitoring
                loss_elementwise = self.criterion(outputs, targets)
                mask_expanded = mask.unsqueeze(-1).expand_as(loss_elementwise)
                masked_loss = (loss_elementwise * mask_expanded).sum()
                valid_elements = mask_expanded.sum()

                running_loss += masked_loss.item()
                total_samples += valid_elements.item()

                # Calculate Distance Errors for Metric
                # Error vector = Predicted Residuals - True Residuals
                # Since True Residuals = GT - Baseline, and Predicted Residuals = Pred - Baseline
                # Error = (Pred - Baseline) - (GT - Baseline) = Pred - GT
                # Distance = sqrt(dN^2 + dE^2)

                # Convert to numpy for metric calculation
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                mask_np = mask.cpu().numpy().astype(bool)

                batch_size = features.shape[0]
                for i in range(batch_size):
                    # Get valid sequence length
                    valid_len = np.sum(mask_np[i])
                    if valid_len == 0:
                        continue

                    # Extract valid predictions and targets
                    pred_seq = outputs_np[i, :valid_len, :]
                    target_seq = targets_np[i, :valid_len, :]

                    # Calculate Euclidean distance error in meters
                    # Shape: (L, 2) -> (L,)
                    diff = pred_seq - target_seq
                    dist_errors = np.sqrt(np.sum(diff**2, axis=1))

                    # Store by phone
                    phone_name = meta_list[i]["phone_name"]
                    phone_errors[phone_name].extend(dist_errors)

        if total_samples == 0:
            print("Warning: Validation set is empty or has no valid samples.")
            return float("nan"), float("inf")

        avg_loss = running_loss / total_samples

        # Compute Competition Metric
        phone_scores = []
        for phone, errors in phone_errors.items():
            if not errors:
                continue
            p50 = np.percentile(errors, 50)
            p95 = np.percentile(errors, 95)
            score = (p50 + p95) / 2
            phone_scores.append(score)

        metric_score = np.mean(phone_scores) if phone_scores else float("inf")

        return avg_loss, metric_score

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on {self.device}...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate()

            # Scheduler step
            self.scheduler.step(val_metric)

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss (MAE): {train_loss:.6f} | "
                f"Val Loss (MAE): {val_loss:.6f} | "
                f"Val Metric (50/95): {val_metric:.10f}"
            )

            # Checkpoint & Early Stopping
            if val_metric < (self.best_score - self.min_delta):
                self.best_score = val_metric
                self.counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> Model saved! Best Score: {self.best_score:.10f}")
            else:
                self.counter += 1
                print(f"  -> No improvement. Counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    self.early_stop = True
                    break

        print("Training complete.")
        print(f"Best Validation Metric: {self.best_score:.10f}")

        # Load best model for inference
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        return self.model


def train_model(train_loader, val_loader, model_class):
    """
    Wrapper function to instantiate model and trainer, then run training.
    """
    model = model_class()
    trainer = Trainer(model, train_loader, val_loader)
    best_model = trainer.fit()
    return best_model
