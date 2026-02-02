import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.architecture import LSA_WDS
from library.data_factory import DataProcessor


# Set fixed random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        atomic_features = batch["atomic_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        global_features = batch["global_features"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_features, batch_indices, global_features)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, batch_indices, global_features)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


class Runner:
    """
    Manages the training and inference lifecycle.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Runner initialized on device: {self.device}")

        # Initialize Model
        self.model = LSA_WDS().to(self.device)

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # Loss function (MSE on log-transformed targets)
        self.criterion = nn.MSELoss()

        # Data Processor
        self.processor = DataProcessor()

    def train(self, load_cached_data=True):
        """
        Runs the full training loop with early stopping.
        """
        print("Starting training process...")

        # Get DataLoaders
        train_loader, val_loader, _ = self.processor.get_dataloaders(
            load_cached_data=load_cached_data
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Train and Validate
            train_loss = train_one_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )
            val_loss = validate(self.model, val_loader, self.criterion, self.device)

            # RMSLE is sqrt of MSE on log-transformed data
            train_rmsle = np.sqrt(train_loss)
            val_rmsle = np.sqrt(val_loss)

            # Scheduler step
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss (MSE): {train_loss:.8f} | "
                f"Val Loss (MSE): {val_loss:.8f} | "
                f"Train RMSLE: {train_rmsle:.8f} | "
                f"Val RMSLE: {val_rmsle:.8f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved with Val RMSLE: {val_rmsle:.8f}")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val RMSLE: {np.sqrt(best_val_loss):.8f}")

    def predict(self, load_cached_data=True):
        """
        Generates predictions on the test set using the best saved model.
        """
        print("Starting inference process...")

        # Load best model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
            )

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        # Get Test DataLoader
        _, _, test_loader = self.processor.get_dataloaders(
            load_cached_data=load_cached_data
        )

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                ids = batch["id"]

                outputs = self.model(atomic_features, batch_indices, global_features)

                # Inverse transform: exp(x) - 1
                # Since targets were log1p transformed
                preds = torch.expm1(outputs)

                all_ids.append(ids.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # Concatenate results
        all_ids = np.concatenate(all_ids)
        all_preds = np.concatenate(all_preds)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": all_ids,
                "formation_energy_ev_natom": all_preds[:, 0],
                "bandgap_energy_ev": all_preds[:, 1],
            }
        )

        # Sort by ID to match sample submission structure usually
        submission_df = submission_df.sort_values("id").reset_index(drop=True)

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print("First 5 rows of submission:")
        print(submission_df.head())
