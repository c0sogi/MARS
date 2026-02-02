import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import VentilatorDataset
from library.model import GraduatedCapacityNetwork
from library.loss import MaskedAuxiliaryLoss


class Trainer:
    """
    Trainer class for the Ventilator Pressure Prediction task.
    Manages data loading, model initialization, training loop, validation, and checkpointing.
    """

    def __init__(self):
        """
        Initialize the Trainer.
        Sets up device, data loaders, model, optimizer, scheduler, and loss function.
        """
        # 1. Setup
        seed_everything(Config.SEED)
        self.device = get_device()
        print(f"Using device: {self.device}")

        # 2. Data Loading
        print("Initializing Datasets...")
        self.train_dataset = VentilatorDataset(split="train", load_cached_data=True)
        self.val_dataset = VentilatorDataset(split="val", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Model Initialization
        # Determine input dimension from dataset features
        # X shape is (N, L, F), so input_dim is F
        input_dim = self.train_dataset.X.shape[-1]
        print(f"Model Input Dimension: {input_dim}")

        self.model = GraduatedCapacityNetwork(input_dim=input_dim)
        self.model.to(self.device)

        # 4. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # OneCycleLR Scheduler
        # Total steps = epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        # 5. Loss Function
        self.criterion = MaskedAuxiliaryLoss(aux_weight=Config.AUX_LOSS_WEIGHT)

        # 6. State
        self.best_val_mae = float("inf")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        # Iterate over batches
        # Using tqdm for visual feedback is allowed, but we keep it minimal as per instructions
        # to "Only print the required information".

        for batch in self.train_loader:
            X, u_out, y, _ = batch

            # Move to device
            X = X.to(self.device)
            u_out = u_out.to(self.device)
            y = y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X)  # returns (final_pred, aux_pred)

            # Compute loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Strict Gradient Clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            # Optimizer Step
            self.optimizer.step()

            # Scheduler Step
            self.scheduler.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Metric: Mean Absolute Error on inspiratory phase (u_out == 0).
        """
        self.model.eval()
        total_mae = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                X, u_out, y, _ = batch

                X = X.to(self.device)
                u_out = u_out.to(self.device)
                y = y.to(self.device)

                # Forward pass (only need final prediction)
                final_pred, _ = self.model(X)

                # Calculate MAE for inspiratory phase
                # Mask: 1 where u_out == 0 (inspiratory), 0 otherwise
                mask = 1.0 - u_out

                mae_sum = torch.sum(torch.abs(final_pred - y) * mask)
                count = torch.sum(mask)

                total_mae += mae_sum.item()
                total_count += count.item()

        # Avoid division by zero
        if total_count == 0:
            return float("inf")

        avg_mae = total_mae / total_count
        return avg_mae

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_mae = self.validate()

            # Print metrics (Full precision)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Save Best Model
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved from {self.best_val_mae} to {val_mae}. Saving model..."
                )
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), Config.MODEL_PATH)

        print(f"Training complete. Best Validation MAE: {self.best_val_mae}")


def run_training():
    """
    Helper function to instantiate Trainer and run the training process.
    """
    trainer = Trainer()
    trainer.fit()
