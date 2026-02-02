import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.dataset import get_dataset
from library.model import DeepDecoupledBiGRU
from library.loss_metric import MCRMSELoss, compute_metric


class Trainer:
    """
    Manages the training lifecycle, including optimization, validation,
    metric tracking, and early stopping.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Objective Function (MCRMSE on all 5 targets)
        self.criterion = MCRMSELoss()

        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # Early Stopping State
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            inputs = batch["inputs"].to(self.device)
            bpp_indices = batch["bpp_indices"].to(self.device)
            bpp_mask = batch["bpp_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, bpp_indices, bpp_mask)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Calculates both the loss and the competition metric (MCRMSE on scored columns).
        """
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                bpp_indices = batch["bpp_indices"].to(self.device)
                bpp_mask = batch["bpp_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(inputs, bpp_indices, bpp_mask)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Collect predictions and targets for global metric calculation
                # Move to CPU immediately to save GPU memory
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Competition Metric
        metric_score = compute_metric(all_preds, all_targets, self.config)

        return avg_loss, metric_score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_metric = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Time: {duration:.2f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MCRMSE: {val_metric}"
            )

            # Early Stopping Check
            if val_metric < self.best_score:
                print(
                    f"Validation improved ({self.best_score} -> {val_metric}). Saving model..."
                )
                self.best_score = val_metric
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.model_save_path)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{self.config.patience}"
                )

                if self.patience_counter >= self.config.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_score}")


def run_training(config=None):
    """
    Entry point to initialize data, model, and trainer, then start training.
    """
    if config is None:
        config = Config()

    # Reproducibility
    set_seed(config.seed)

    # 1. Load Data
    # Note: get_dataset handles caching internally via load_or_process_data
    print("Initializing datasets...")
    train_dataset = get_dataset("train", config)
    val_dataset = get_dataset("val", config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 2. Initialize Model
    print("Initializing model...")
    model = DeepDecoupledBiGRU(config)

    # 3. Initialize Trainer and Fit
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.fit()
