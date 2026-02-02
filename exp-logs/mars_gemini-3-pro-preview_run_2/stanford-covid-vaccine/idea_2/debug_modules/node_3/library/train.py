import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_dataloaders
from library.model import GEHN


class Trainer:
    """
    Trainer class for the GEHN model.
    Handles training, validation, and model persistence.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Initialize Model
        self.model = GEHN(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss Function
        # We optimize on all 5 targets, but only for the first 68 positions (pred_len)
        # The MCRMSELoss class handles the slicing internally.
        self.criterion = MCRMSELoss(num_scored=config.pred_len)

        # Identify indices of the columns that are officially scored in the competition
        # Target cols: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Scored cols: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i for i, col in enumerate(config.target_cols) if col in config.scored_cols
        ]

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        Returns the average batch loss.
        """
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            adj = batch["adj"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(inputs, adj)

            # Loss is calculated on all 5 targets
            loss = self.criterion(preds, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(loader.dataset)

    def validate_epoch(self, loader):
        """
        Runs validation.
        Calculates the EXACT MCRMSE over the whole dataset (not average of batch averages).
        Returns:
            val_loss: MCRMSE on all 5 targets (optimization objective)
            val_comp_metric: MCRMSE on the 3 scored targets (competition metric)
        """
        self.model.eval()

        # Accumulators for exact MCRMSE calculation
        # Shape: (Num_Targets,)
        total_squared_error = torch.zeros(self.config.num_targets, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                adj = batch["adj"].to(self.device)
                targets = batch["targets"].to(self.device)

                preds = self.model(inputs, adj)

                # Slice to scored length (68)
                preds_scored = preds[:, : self.config.pred_len, :]
                targets_scored = targets[:, : self.config.pred_len, :]

                # Sum of squared errors per column for this batch
                # Sum over batch (dim 0) and sequence (dim 1)
                batch_se = torch.sum((preds_scored - targets_scored) ** 2, dim=(0, 1))

                total_squared_error += batch_se

                # Count total elements per column: Batch_Size * Scored_Seq_Len
                total_samples += inputs.size(0) * self.config.pred_len

        # Calculate MSE per column
        mse_per_col = total_squared_error / total_samples

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 1. Loss (All 5 targets) - Average RMSE across all 5 columns
        val_loss = torch.mean(rmse_per_col).item()

        # 2. Competition Metric (Specific 3 targets) - Average RMSE across scored columns
        scored_rmse = rmse_per_col[self.scored_indices]
        val_comp_metric = torch.mean(scored_rmse).item()

        return val_loss, val_comp_metric

    def fit(self, train_loader, val_loader):
        """
        Main training loop with early stopping and logging.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")
        print(f"Optimization Target: MCRMSE on all {self.config.num_targets} columns.")
        print(f"Reporting Metric: MCRMSE on {len(self.scored_indices)} scored columns.")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_comp_metric = self.validate_epoch(val_loader)

            # Step scheduler based on validation loss
            self.scheduler.step(val_loss)

            duration = time.time() - start_time

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss (All 5): {val_loss} | "
                f"Val Metric (Scored 3): {val_comp_metric}"
            )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.best_model_path)
                print(f"New best model saved to {self.config.best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val Loss: {best_val_loss}")


def train_main():
    """
    Main function to setup and run training.
    """
    # 1. Setup
    seed_everything(42)
    config = Config()

    # 2. Data
    # load_cached_data=True ensures we use the cache if available,
    # or create it using metadata if not.
    train_loader, val_loader, _ = get_dataloaders(config, load_cached_data=True)

    # 3. Training
    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)
