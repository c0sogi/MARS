import os
import time
import torch
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_dataloaders
from library.model import DenseContextNet


class Trainer:
    """
    Manages the training and validation lifecycle.
    """

    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, loader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, partner_indices, targets in loader:
            # Move to device
            inputs = inputs.to(self.device)
            partner_indices = partner_indices.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, partner_indices)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, loader):
        """
        Performs validation using the MetricTracker for correct global MCRMSE.
        """
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for inputs, partner_indices, targets in loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)

                # Forward pass
                outputs = self.model(inputs, partner_indices)

                # Update tracker (handles CPU/Numpy conversion internally)
                tracker.update(outputs, targets)

        return tracker.result()

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Main training loop with early stopping and model saving.
        """
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_mcrmse = self.validate(val_loader)

            # Scheduler Step
            # ReduceLROnPlateau expects the metric to minimize
            self.scheduler.step(val_mcrmse)

            elapsed = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val MCRMSE: {val_mcrmse}"
            )

            # Save Best Model & Early Stopping
            if val_mcrmse < best_score:
                print(
                    f"Validation score improved ({best_score} -> {val_mcrmse}). Saving model..."
                )
                best_score = val_mcrmse
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Val MCRMSE: {best_score}")


def train_model(debug=False, batch_size=None, epochs=None):
    """
    Initializes components and runs the training process.

    Args:
        debug (bool): If True, runs in debug mode (handled by Config setup logic if applicable,
                      or overrides here).
        batch_size (int, optional): Override default batch size.
        epochs (int, optional): Override default number of epochs.
    """
    # 1. Setup Configuration
    Config.setup()
    set_seed(Config.SEED)

    # Apply Overrides
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size
    if epochs is not None:
        Config.EPOCHS = epochs

    if debug:
        # Enforce debug settings
        Config.EPOCHS = 2
        Config.BATCH_SIZE = 4
        print("Debug mode enabled: EPOCHS=2, BATCH_SIZE=4")

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, _ = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DenseContextNet().to(device)

    # 4. Loss Function
    criterion = MaskedMCRMSELoss().to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 6. Trainer Initialization
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # 7. Start Training
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )
