import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import MoleculeGraphDataset, GraphCollate
from library.model import MPDIN
from library.utils import GroupStandardizer, LogMAE


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)
        Config.setup_directories()

        # --- Data Preparation ---
        print("Initializing DataLoaders...")
        self.collate_fn = GraphCollate(device=self.device)

        # Train Loader
        train_dataset = MoleculeGraphDataset(split="train", load_cached_data=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        # Validation Loader
        val_dataset = MoleculeGraphDataset(split="val", load_cached_data=True)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        # Test Loader (initialized later for prediction to save memory if needed,
        # but can be done here)
        self.test_dataset = MoleculeGraphDataset(split="test", load_cached_data=True)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        # --- Model & Optimization ---
        print("Initializing Model...")
        self.model = MPDIN().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=Config.SCHEDULER_T_0,
            T_mult=Config.SCHEDULER_T_MULT,
            eta_min=Config.SCHEDULER_ETA_MIN,
        )

        self.criterion = nn.L1Loss()
        self.standardizer = GroupStandardizer(device=self.device)

        # Training State
        self.best_val_metric = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch in self.train_loader:
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            targets = batch["coupling_value"]
            types = batch["coupling_type"]

            # Standardize targets
            # z = (y - mean) / std
            standardized_targets = self.standardizer.transform(targets, types)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(batch)

            # Compute Loss
            loss = self.criterion(preds, standardized_targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        duration = time.time() - start_time

        # Step scheduler at the end of epoch
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
            f"Train Loss (L1): {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {duration:.1f}s"
        )
        return avg_loss

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Forward pass
                preds_z = self.model(batch)

                # Inverse Transform: y_hat = z_hat * std + mean
                types = batch["coupling_type"]
                preds_original = self.standardizer.inverse_transform(preds_z, types)

                targets_original = batch["coupling_value"]

                all_preds.append(preds_original)
                all_targets.append(targets_original)
                all_types.append(types)

        # Concatenate for full metric calculation
        if not all_preds:
            return float("inf")

        full_preds = torch.cat(all_preds)
        full_targets = torch.cat(all_targets)
        full_types = torch.cat(all_types)

        # Compute LogMAE on original scale
        val_metric = LogMAE.compute(full_preds, full_targets, full_types)

        return val_metric.item()

    def fit(self):
        print("\nStarting Training...")

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            _ = self.train_epoch(epoch)

            val_metric = self.validate()
            print(f"Validation LogMAE: {val_metric}")  # Full precision print

            # Checkpointing
            if val_metric < self.best_val_metric:
                print(
                    f"Metric improved ({self.best_val_metric:.6f} -> {val_metric:.6f}). Saving model..."
                )
                self.best_val_metric = val_metric
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"\nTraining Complete. Best Validation LogMAE: {self.best_val_metric}")

    def predict_and_submit(self):
        print("\nGenerating Submission...")

        # Load Best Model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No model checkpoint found. Skipping submission.")
            return

        print(f"Loading model from {Config.MODEL_SAVE_PATH}")
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Forward pass
                preds_z = self.model(batch)

                # Inverse Transform
                types = batch["coupling_type"]
                preds_original = self.standardizer.inverse_transform(preds_z, types)

                all_ids.append(batch["coupling_id"].cpu().numpy())
                all_preds.append(preds_original.cpu().numpy())

        # Concatenate
        if all_ids:
            final_ids = np.concatenate(all_ids)
            final_preds = np.concatenate(all_preds)
        else:
            final_ids = np.array([])
            final_preds = np.array([])

        # Create DataFrame
        df_sub = pd.DataFrame(
            {"id": final_ids, "scalar_coupling_constant": final_preds}
        )

        # Sort by ID to match sample submission order (good practice)
        df_sub = df_sub.sort_values("id")

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")

    def run(self):
        self.fit()
        self.predict_and_submit()
