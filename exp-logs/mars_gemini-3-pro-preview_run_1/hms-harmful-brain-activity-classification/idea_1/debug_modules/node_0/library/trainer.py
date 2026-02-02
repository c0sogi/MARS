import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import (
    DEVICE,
    CACHE_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    set_seed,
)
from library.utils import compute_kl_divergence


class Trainer:
    """
    Manages the training, validation, and optimization of the SpectrogramCRNN model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = DEVICE,
    ):
        """
        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Compute device ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # KLDivLoss expects log-probabilities as input and probabilities as target.
        # reduction='batchmean' aligns mathematically with the KL definition.
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        # Ensure cache directory exists for saving models
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.save_path = os.path.join(CACHE_DIR, "best_model.pth")

    def train_epoch(self, optimizer):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = self.model(data)

            # KLDivLoss requires Log-Probabilities
            # Add epsilon for numerical stability before log
            log_probs = torch.log(outputs + 1e-10)

            loss = self.criterion(log_probs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            count += data.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns average loss (for scheduler) and KL metric (for reporting).
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in self.val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)

                # Compute Loss
                log_probs = torch.log(outputs + 1e-10)
                loss = self.criterion(log_probs, targets)

                running_loss += loss.item() * data.size(0)
                count += data.size(0)

                # Store for Metric Calculation
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        # Compute exact competition metric
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        metric_score = compute_kl_divergence(all_targets, all_preds)

        return avg_loss, metric_score

    def fit(self, epochs=EPOCHS, lr=LEARNING_RATE):
        """
        Runs the full training pipeline with Early Stopping and LR Scheduling.
        """
        print(f"Starting training on {self.device} for {epochs} epochs...")

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(optimizer)

            # Validate
            val_loss, val_metric = self.validate()

            # Update Scheduler
            scheduler.step(val_loss)

            # Print Metrics
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Metric (KL): {val_metric}"
            )

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                # print(f"  -> Model saved to {self.save_path}")
            else:
                patience_counter += 1
                # print(f"  -> No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
        print(f"Best Validation Loss: {best_val_loss}")
