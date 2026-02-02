import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.utils import seed_everything, calc_log_mae, Standardizer
from library.data import MoleculeDataset, collate_molecules
from library.model import MPDIN


class Trainer:
    """
    Manages the training, validation, and saving of the MP-DIN model.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        # Set seeds for reproducibility
        seed_everything(config.seed)

        # Initialize Model
        self.model = MPDIN(config).to(self.device)

        # Initialize Standardizer (loads stats or fits if needed)
        self.standardizer = Standardizer(config)
        # We assume standardizer is already fit on train data via data preprocessing
        # but we ensure it's loaded here.
        self.standardizer.load()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        # T_0 is set to T_max (epochs) for a single cycle, or can be tuned.
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.T_max, T_mult=1, eta_min=config.eta_min
        )

        # Loss Function (L1 Loss for robustness)
        self.criterion = nn.L1Loss()

    def fit(self):
        """
        Executes the training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        # Data Loaders
        train_dataset = MoleculeDataset(
            self.config, split="train", load_cached_data=True
        )
        val_dataset = MoleculeDataset(self.config, split="val", load_cached_data=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_molecules,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_molecules,
            pin_memory=True,
        )

        best_val_metric = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # Training Step
            train_loss = self.train_epoch(train_loader)

            # Validation Step
            val_loss, val_metric = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val LogMAE: {val_metric:.20f}"
            )

            # Early Stopping & Checkpointing
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved! LogMAE: {best_val_metric:.20f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Runs one epoch of training.
        Returns average training loss.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch)

            # Get targets and standardize them
            targets = batch["coupling_value"].view(-1, 1)
            types = batch["coupling_type"]

            # Standardize targets for stable training
            targets_std = self.standardizer.transform(targets.squeeze(), types).view(
                -1, 1
            )

            # Compute Loss
            loss = self.criterion(preds, targets_std)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)

        return total_loss / count if count > 0 else 0.0

    def validate(self, loader: DataLoader):
        """
        Runs validation.
        Returns (average val loss, LogMAE metric).
        """
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_targets = []
        all_types = []

        count = 0

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                preds_std = self.model(batch)

                targets = batch["coupling_value"].view(-1, 1)
                types = batch["coupling_type"]

                # Standardize targets for loss calculation
                targets_std = self.standardizer.transform(
                    targets.squeeze(), types
                ).view(-1, 1)

                loss = self.criterion(preds_std, targets_std)
                total_loss += loss.item() * targets.size(0)
                count += targets.size(0)

                # Inverse transform predictions for metric calculation
                preds_original = self.standardizer.inverse_transform(
                    preds_std.squeeze(), types
                )

                all_preds.append(preds_original)
                all_targets.append(targets.squeeze())
                all_types.append(types)

        avg_loss = total_loss / count if count > 0 else 0.0

        # Concatenate all batches for metric calculation
        if len(all_preds) > 0:
            y_pred = torch.cat(all_preds)
            y_true = torch.cat(all_targets)
            t_types = torch.cat(all_types)

            metric = calc_log_mae(y_true, y_pred, t_types)
        else:
            metric = 0.0

        return avg_loss, metric


def predict_submission(config: Config):
    """
    Loads the best model, generates predictions for the test set,
    and saves the submission file.
    """
    print("Generating submission...")

    # Load Model
    device = torch.device(config.device)
    model = MPDIN(config).to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Standardizer
    standardizer = Standardizer(config)
    standardizer.load()

    # Data Loader
    test_dataset = MoleculeDataset(config, split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            preds_std = model(batch)

            # Get types for inverse transform
            types = batch["coupling_type"]
            ids = batch["coupling_id"]

            # Inverse transform
            preds_original = standardizer.inverse_transform(preds_std.squeeze(), types)

            ids_list.append(ids.cpu().numpy())
            preds_list.append(preds_original.cpu().numpy())

    # Concatenate results
    if len(ids_list) > 0:
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)
    else:
        all_ids = np.array([], dtype=np.int32)
        all_preds = np.array([], dtype=np.float32)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

    # Sort by ID to ensure correct order
    df_sub.sort_values("id", inplace=True)

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
