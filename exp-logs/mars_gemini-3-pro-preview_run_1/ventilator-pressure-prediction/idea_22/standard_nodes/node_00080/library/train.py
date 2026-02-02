import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from library.config import Config
from library.utils import seed_everything, compute_metric, get_device
from library.dataset import get_data_loaders
from library.model import WideProjectedNet
from library.loss import MaskedL1Loss


class Trainer:
    """
    Trainer class to manage the training and validation of the Wide-Projected Network.
    Encapsulates the training loop, validation logic, and model checkpointing.
    """

    def __init__(
        self, model, device, criterion, optimizer, scheduler=None, config=Config
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.best_val_mae = float("inf")

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Unpack data and move to device
            x = batch["X"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Returns tuple: (final_pred, aux_pred)
            preds = self.model(x)

            # Compute loss (MaskedL1Loss handles auxiliary weighting and masking)
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Strict stability enforcement for LSTM)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            )

            # Optimization Step
            self.optimizer.step()

            # Scheduler Step (OneCycleLR updates per step)
            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation and computes MAE on the inspiratory phase.
        """
        self.model.eval()
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                # Forward pass (only need final prediction for metric)
                final_pred, _ = self.model(x)

                # Squeeze final_pred from (B, S, 1) to (B, S) to match y and u_out
                final_pred = final_pred.squeeze(-1)

                # Compute Metric (MAE on inspiratory phase)
                # compute_metric expects detached tensors and handles masking internally
                mae = compute_metric(final_pred, y, u_out)

                total_mae += mae
                count += 1

        # Return average MAE across batches
        return total_mae / count if count > 0 else 0.0

    def fit(self, train_loader, val_loader, epochs, patience=10):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_mae = self.validate(val_loader)

            # Timing
            epoch_time = time.time() - start_time

            # Logging (Full precision metrics)
            print(
                f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Checkpointing
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved from {self.best_val_mae} to {val_mae}. Saving model..."
                )
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        print(f"Training complete. Best Val MAE: {self.best_val_mae}")


def train_model(
    load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
):
    """
    Orchestrates the training process:
    1. Sets seeds for reproducibility.
    2. Loads data (cached or computed).
    3. Initializes model, optimizer, scheduler, and loss.
    4. Runs the training loop.

    Args:
        load_cached_data (bool): Whether to use cached feature arrays.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_data_loaders(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
    )

    # Determine input dimension from the dataset dynamically
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["X"].shape[-1]
    print(f"Detected Input Dimension: {input_dim}")

    # 3. Model Initialization
    model = WideProjectedNet(input_dim=input_dim).to(device)

    # 4. Loss Function
    criterion = MaskedL1Loss()

    # 5. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # 6. Scheduler (OneCycleLR)
    # Note: steps_per_epoch is required for OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # 7. Trainer Initialization
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=Config,
    )

    # 8. Run Training
    # Using a default patience of 10 epochs for early stopping
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=10)

    return trainer
