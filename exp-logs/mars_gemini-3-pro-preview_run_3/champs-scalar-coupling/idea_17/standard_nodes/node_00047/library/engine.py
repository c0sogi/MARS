import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import MetricLogger, TargetStandardizer, set_seed
from library.model import MPDCFN
from library.dataset import MoleculeDataset, collate_molecular_graphs


class Trainer:
    """
    Manages the training lifecycle of the MP-DCFN model.
    Handles data loading, training loops, validation, metric calculation,
    optimization, and checkpointing.
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        print("Initializing model...")
        self.model = MPDCFN().to(self.device)

        # Initialize Target Standardizer (loads stats from cache)
        self.standardizer = TargetStandardizer(device=self.device)
        # We assume preprocessing has run and cache exists, so we load it.
        # If not, fit() would need a dataframe, but here we rely on the pipeline order.
        self.standardizer.fit(df=None, load_cached_data=True)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.SCHEDULER_T_0, T_mult=Config.SCHEDULER_T_MULT
        )

        # Loss Function (L1 Loss / MAE)
        self.criterion = nn.L1Loss()

        # Metric Logger
        self.metric_logger = MetricLogger()

    def train_epoch(self, dataloader, epoch_idx):
        """
        Executes one training epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch in dataloader:
            # Move batch to device
            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch[key] = val.to(self.device)

            # Get targets and types
            targets_raw = batch["coupling_value"]
            types = batch["coupling_type"]

            # Standardize targets for training stability
            # z = (y - mu) / sigma
            targets_std = self.standardizer.transform(targets_raw, types)

            # Forward Pass
            self.optimizer.zero_grad()
            preds_std = self.model(batch)

            # Compute Loss
            loss = self.criterion(preds_std, targets_std)

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1} | Train Loss (L1 Std): {avg_loss:.6f} | Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Computes the Log Mean Absolute Error (LMAE) on the original scale.
        """
        self.model.eval()
        self.metric_logger.reset()

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key, val in batch.items():
                    if torch.is_tensor(val):
                        batch[key] = val.to(self.device)

                # Get targets and types
                targets_raw = batch["coupling_value"]
                types = batch["coupling_type"]

                # Forward Pass
                preds_std = self.model(batch)

                # Inverse Transform Predictions to Physical Scale
                # y_pred = z_pred * sigma + mu
                preds_real = self.standardizer.inverse_transform(preds_std, types)

                # Update Metric Logger
                self.metric_logger.update(preds_real, targets_raw, types)

        # Compute final metrics
        metrics = self.metric_logger.compute()
        return metrics

    def fit(self):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        print("Loading datasets...")
        train_dataset = MoleculeDataset(split="train")
        val_dataset = MoleculeDataset(split="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_molecular_graphs,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_molecular_graphs,
            pin_memory=True,
        )

        print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

        best_lmae = float("inf")
        patience_counter = 0

        for epoch in range(Config.MAX_EPOCHS):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)
            val_lmae = val_metrics["LMAE"]

            # Print Validation Results
            print(f"Epoch {epoch+1} | Val LMAE: {val_lmae}")
            # Optional: Print per-type details if needed, but keeping it concise
            # print(f"  Per Type: {val_metrics['per_type']}")

            # Scheduler Step
            self.scheduler.step()

            # Checkpoint & Early Stopping
            if val_lmae < best_lmae:
                best_lmae = val_lmae
                patience_counter = 0
                print(f"  New best model found! Saving to {Config.BEST_MODEL_PATH}...")
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation LMAE: {best_lmae}")
