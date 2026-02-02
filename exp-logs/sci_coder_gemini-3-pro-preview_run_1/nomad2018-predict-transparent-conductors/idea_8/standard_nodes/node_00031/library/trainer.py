import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from library.config import Config
from library.utils import get_logger, set_seed
from library.data_loader import get_dataloaders
from library.model import SIRDS_SP

# Initialize logger
logger = get_logger("trainer")


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the SI-RDS-SP model.
    """

    def __init__(self, model, device=None):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device, optional): The device to run on. Defaults to Config.DEVICE.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = model.to(self.device)

        # Loss function: MSE on log-transformed targets corresponds to MSLE
        self.criterion = nn.MSELoss()

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Reduce LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            atomic_x = batch["atomic_x"].to(self.device)
            atomic_mask = batch["atomic_mask"].to(self.device)
            global_x = batch["global_x"].to(self.device)
            symmetry_x = batch["symmetry_x"].to(self.device)
            targets = batch["y"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(atomic_x, atomic_mask, global_x, symmetry_x)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_x = batch["atomic_x"].to(self.device)
                atomic_mask = batch["atomic_mask"].to(self.device)
                global_x = batch["global_x"].to(self.device)
                symmetry_x = batch["symmetry_x"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(atomic_x, atomic_mask, global_x, symmetry_x)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        return val_loss

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    ):
        """
        Runs the full training loop with early stopping.
        """
        logger.info(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # RMSLE is approx sqrt(MSE) since targets are log-transformed
            train_rmsle = np.sqrt(train_loss)
            val_rmsle = np.sqrt(val_loss)

            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f} (RMSLE: {train_rmsle:.6f}) - "
                f"Val Loss: {val_loss:.6f} (RMSLE: {val_rmsle:.6f})"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                logger.info(f"  -> New best model saved! Val Loss: {val_loss:.8f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.2f} seconds. Best Val Loss: {best_val_loss:.8f}"
        )

    def predict(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        """
        logger.info("Loading best model for prediction...")
        if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            logger.error("No checkpoint found! Cannot predict.")
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        logger.info("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["id"]
                atomic_x = batch["atomic_x"].to(self.device)
                atomic_mask = batch["atomic_mask"].to(self.device)
                global_x = batch["global_x"].to(self.device)
                symmetry_x = batch["symmetry_x"].to(self.device)

                # Forward pass
                outputs = self.model(atomic_x, atomic_mask, global_x, symmetry_x)

                # Inverse transform: exp(y) - 1
                # Targets were trained on log1p(y)
                preds = torch.expm1(outputs)

                all_ids.extend(ids)
                all_preds.append(preds.cpu().numpy())

        # Concatenate all predictions
        all_preds = np.vstack(all_preds)

        # Create DataFrame
        submission_df = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
        submission_df.insert(0, "id", all_ids)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        logger.info(f"First 5 predictions:\n{submission_df.head()}")


def run_training(load_cached_data=True):
    """
    Main entry point to run the training pipeline.
    """
    set_seed(Config.SEED)

    # 1. Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = SIRDS_SP()

    # 3. Initialize Trainer
    trainer = Trainer(model)

    # 4. Train
    trainer.fit(train_loader, val_loader)

    # 5. Predict
    trainer.predict(test_loader)
