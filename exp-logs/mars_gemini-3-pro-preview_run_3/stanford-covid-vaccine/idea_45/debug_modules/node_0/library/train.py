import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_loader
from library.model import DSDBiGRUModel


class Trainer:
    """
    Manages the training, validation, and saving of the RNA degradation model.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = DSDBiGRUModel(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )

        # Loss Function
        # We use MSELoss. Since MCRMSE is root mean squared error,
        # minimizing MSE is equivalent for optimization purposes.
        self.criterion = nn.MSELoss()

        # Early Stopping State
        self.best_val_score = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(self.device)
            bpp_indices = batch["bpp_indices"].to(self.device)
            bpp_mask = batch["bpp_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, bpp_indices, bpp_mask)

            # Slice to scored length (first 68 positions) for loss calculation
            # as per strategy to focus optimization on scored regions.
            outputs_scored = outputs[:, : self.config.PRED_LEN, :]
            targets_scored = targets[:, : self.config.PRED_LEN, :]

            loss = self.criterion(outputs_scored, targets_scored)

            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            )

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                bpp_indices = batch["bpp_indices"].to(self.device)
                bpp_mask = batch["bpp_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(inputs, bpp_indices, bpp_mask)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metric using the utility function
        # This handles slicing and column selection internally
        score = mcrmse_metric(all_preds, all_targets)
        return score

    def run(self):
        set_seed(self.config.SEED)

        print(f"Initializing training on device: {self.device}")

        # Data Loaders
        train_loader = get_loader(
            "train",
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            shuffle=True,
        )

        val_loader = get_loader(
            "val",
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            shuffle=False,
        )

        print("Starting training loop...")

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | "
                f"Val MCRMSE: {val_score}"
            )

            # Early Stopping and Model Saving
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                print(f"New best model found! Saving to {self.config.BEST_MODEL_PATH}")
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{self.config.PATIENCE}"
                )

            if self.patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MCRMSE: {self.best_val_score}")


def train_model():
    """
    Wrapper function to instantiate the Trainer and run the training process.
    """
    trainer = Trainer()
    trainer.run()
