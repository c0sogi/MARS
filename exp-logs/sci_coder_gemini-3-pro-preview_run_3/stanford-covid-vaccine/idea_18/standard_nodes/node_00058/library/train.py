import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RISRBiGRU
from library.loss import MCRMSELoss


class Trainer:
    """
    Trainer class for the RNA Degradation Prediction task.
    Manages the training loop, validation, optimization, and checkpointing.
    """

    def __init__(self, config: Config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Initialize Model
        self.model = RISRBiGRU(config).to(self.device)

        # Initialize Loss
        self.criterion = MCRMSELoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )

        # Tracking best performance
        self.best_score = float("inf")

        # Determine indices for scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # config.target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # config.scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i
            for i, col in enumerate(config.target_cols)
            if col in config.scored_targets
        ]

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, adjacency, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            adjacency = adjacency.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(inputs, adjacency)

            # Calculate Loss
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Runs validation and calculates the global MCRMSE metric.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, adjacency, targets in self.val_loader:
                inputs = inputs.to(self.device)
                adjacency = adjacency.to(self.device)

                # Forward pass
                preds = self.model(inputs, adjacency)

                # Slice predictions to match target length if necessary
                if preds.shape[1] > targets.shape[1]:
                    preds = preds[:, : targets.shape[1], :]

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE on all 5 columns (matches Loss)
        val_loss_metric = calculate_mcrmse(all_preds, all_targets)

        # Calculate MCRMSE on Scored Columns (Competition Metric)
        val_score = calculate_mcrmse(
            all_preds, all_targets, scored_indices=self.scored_indices
        )

        return val_loss_metric, val_score

    def fit(self):
        """
        Main training loop with early stopping and checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        print(
            f"Config: Epochs={self.config.epochs}, Batch Size={self.config.batch_size}, LR={self.config.lr}"
        )

        patience = 5
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss (All): {val_loss:.6f} | "
                f"Val Score (Scored): {val_score} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing based on Scored Columns Metric
            if val_score < self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.config.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Val Score: {self.best_score}")


def run_training(config: Config = None):
    """
    Entry point to run the training process.

    Args:
        config (Config, optional): Configuration object. If None, creates a default one.
    """
    if config is None:
        config = Config()

    # Reproducibility
    seed_everything(config.seed)

    # Data Loading
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config, load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer(config, train_loader, val_loader)

    # Start Training
    trainer.fit()
