import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RNADataset
from library.model import LatentSpatialBiGRU
from library.loss import MCRMSELoss


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Trainer class for the RNA degradation prediction model.
    Handles training loops, validation, early stopping, and model saving.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else Config()

        # Set device
        self.device = torch.device(self.config.device)

        # Initialize Model
        self.model = LatentSpatialBiGRU(self.config).to(self.device)

        # Initialize Loss
        self.criterion = MCRMSELoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.T_max, eta_min=self.config.eta_min
        )

        # Determine indices for scored columns
        # target_cols: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # scored_cols: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i
            for i, col in enumerate(self.config.target_cols)
            if col in self.config.scored_cols
        ]

        # Ensure working directory exists for saving models
        os.makedirs(self.config.working_dir, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(self.device)
            pair_indices = batch["pair_index"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, pair_indices)

            # Compute loss
            # MCRMSELoss handles slicing internally if shapes mismatch,
            # but we pass full tensors here.
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the full validation set.
        Aggregates predictions globally before calculating metrics.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(self.device)
                pair_indices = batch["pair_index"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass
                outputs = self.model(inputs, pair_indices)

                # Slice outputs to match target length (seq_scored=68)
                # Targets are already (Batch, 68, 5)
                # Outputs are (Batch, 107, 5)
                seq_len_target = targets.shape[1]
                if outputs.shape[1] > seq_len_target:
                    outputs = outputs[:, :seq_len_target, :]

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Global Aggregation
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE for all columns (Training Objective)
        # We can reuse the criterion class logic or compute manually
        mse_all = torch.mean((all_preds - all_targets) ** 2, dim=(0, 1))
        rmse_all = torch.sqrt(mse_all)
        mcrmse_all = torch.mean(rmse_all).item()

        # Calculate MCRMSE for Scored Columns (Competition Metric)
        scored_preds = all_preds[:, :, self.scored_indices]
        scored_targets = all_targets[:, :, self.scored_indices]

        mse_scored = torch.mean((scored_preds - scored_targets) ** 2, dim=(0, 1))
        rmse_scored = torch.sqrt(mse_scored)
        mcrmse_scored = torch.mean(rmse_scored).item()

        return mcrmse_all, mcrmse_scored

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        set_seed(self.config.seed)

        print("Loading Datasets...")
        train_dataset = RNADataset(split="train", load_cached_data=load_cached_data)
        val_dataset = RNADataset(split="val", load_cached_data=load_cached_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        print(f"Starting training on {self.device}...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss_all, val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE (All): {val_loss_all} | "
                f"Val MCRMSE (Scored): {val_score}"
            )

            # Early Stopping Logic based on Scored Columns
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"  >>> New Best Model Saved! Score: {best_score}")
            else:
                patience_counter += 1
                print(
                    f"  >>> No improvement. Patience: {patience_counter}/{self.config.patience}"
                )

            if patience_counter >= self.config.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score}")
