import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, compute_rmsle, TargetScaler
from library.data import get_dataloaders
from library.model import kRACGN


class Trainer:
    """
    Trainer class for the k-RA-CGN model.
    """

    def __init__(self, model, train_loader, val_loader, scaler, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.config = config
        self.device = config.DEVICE

        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self):
        """
        Runs one epoch of training.
        Returns:
            float: Average training loss (MSE on standardized targets).
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)

            # Standardize targets
            targets_np = batch.y.cpu().numpy()
            targets_scaled_np = self.scaler.transform(targets_np)
            targets_scaled = torch.tensor(
                targets_scaled_np, dtype=torch.float32, device=self.device
            )

            # Compute loss
            loss = self.criterion(outputs, targets_scaled)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(self.train_loader.dataset)

    def evaluate(self, loader):
        """
        Evaluates the model on a given loader.
        Returns:
            float: Average loss (MSE on standardized targets).
            float: RMSLE on original scale.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(batch)

                # Targets
                targets_np = batch.y.cpu().numpy()
                targets_scaled_np = self.scaler.transform(targets_np)
                targets_scaled = torch.tensor(
                    targets_scaled_np, dtype=torch.float32, device=self.device
                )

                # Loss calculation (Standardized)
                loss = self.criterion(outputs, targets_scaled)
                total_loss += loss.item() * batch.num_graphs

                # Inverse transform predictions for metrics
                preds_np = outputs.cpu().numpy()
                preds_original = self.scaler.inverse_transform(preds_np)

                all_preds.append(preds_original)
                all_targets.append(targets_np)

        avg_loss = total_loss / len(loader.dataset)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute RMSLE on original scale
        rmsle = compute_rmsle(all_targets, all_preds)

        return avg_loss, rmsle

    def run(self):
        """
        Runs the full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss, val_rmsle = self.evaluate(self.val_loader)

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val RMSLE: {val_rmsle}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_PATH)
                print(f"New best model saved to {self.config.CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break


def run_training(load_cached_data=True):
    """
    Main function to execute the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed graph data from cache.
    """
    # 1. Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Prepare Data
    train_loader, val_loader, _, scaler = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Initialize Model
    model = kRACGN()

    # 4. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, scaler, Config)

    # 5. Run Training
    trainer.run()
