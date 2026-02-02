import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, GlobalMetrics
from library.loss import MaskedMCRMSELoss
from library.data import RNADataset
from library.model import DDFRN


class Trainer:
    """
    Manages the training, validation, and checkpointing of the DDFRN model.
    """

    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_score = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(self.device)
            partner_indices = batch["partner_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: Get both pass 1 (y1) and pass 2 (y2) predictions
            y1, y2 = self.model(inputs, partner_indices)

            # Compute Loss: L_total = L(y2) + 0.5 * L(y1)
            loss_y2 = self.criterion(y2, targets)
            loss_y1 = self.criterion(y1, targets)
            loss = loss_y2 + (Config.AUX_LOSS_WEIGHT * loss_y1)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Runs validation using GlobalMetrics to ensure correct MCRMSE calculation.
        """
        self.model.eval()
        metric_calculator = GlobalMetrics()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                partner_indices = batch["partner_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward pass: We only care about the refined prediction y2 for scoring
                _, y2 = self.model(inputs, partner_indices)

                # Update global metrics accumulator
                metric_calculator.update(targets, y2)

        # Compute final metric over the entire validation set
        final_mcrmse = metric_calculator.compute()
        return final_mcrmse

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Main training loop with Early Stopping.
        """
        patience_counter = 0

        print(f"Starting training on device: {self.device}")
        print(
            f"Epochs: {epochs}, Patience: {patience}, Batch Size: {Config.BATCH_SIZE}"
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_score = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_score)
                else:
                    self.scheduler.step()

            # Logging (Full precision for validation score)
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
            )

            # Early Stopping and Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> New Best Model Saved (Score: {self.best_score})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_score}")


def run_training(debug=False):
    """
    Orchestrates the data loading, model initialization, and training process.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    # Debug mode limits the dataset size
    debug_subset = Config.DEBUG_SUBSET_SIZE if (debug or Config.DEBUG) else None

    print("Initializing Datasets...")
    train_dataset = RNADataset(
        mode="train", load_cached_data=True, debug_subset=debug_subset
    )
    val_dataset = RNADataset(
        mode="val", load_cached_data=True, debug_subset=debug_subset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing DDFRN Model...")
    model = DDFRN().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    criterion = MaskedMCRMSELoss()

    # 5. Trainer Initialization and Fitting
    trainer = Trainer(model, device, criterion, optimizer, scheduler)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    return trainer.best_score
