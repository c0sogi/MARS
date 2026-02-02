import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import get_device, compute_mae, seed_everything
from library.model import FMDHNet
from library.dataset import prepare_data


class Trainer:
    """
    Trainer class for the FMDH-Net model.
    Handles training, validation, checkpointing, and inference.
    """

    def __init__(self, model):
        self.device = get_device()
        self.model = model.to(self.device)

        # Optimizer: AdamW is generally robust for this type of regression
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Reduce LR when validation MAE plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss Function: L1 Loss (MAE)
        # reduction='none' allows us to apply the mask manually before averaging
        self.criterion = nn.L1Loss(reduction="none")

        self.best_val_mae = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        Returns the average masked L1 loss.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # Extract u_out for masking.
            # Based on Config.FEATURE_COLS, u_out is at index 2.
            # x shape: (Batch, Seq_Len, Features)
            u_out = x[:, :, 2]

            # Create mask for inspiratory phase (u_out == 0)
            mask = u_out == 0

            self.optimizer.zero_grad()

            # Forward pass
            # Model output: (Batch, Seq_Len, 1) -> Squeeze to (Batch, Seq_Len)
            preds = self.model(x).squeeze(-1)

            # Apply Mask: Select only inspiratory phase predictions and targets
            masked_preds = preds[mask]
            masked_targets = y[mask]

            # Compute Loss only if there are valid samples in the batch
            if masked_targets.numel() > 0:
                loss = self.criterion(masked_preds, masked_targets).mean()

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def validate(self, val_loader):
        """
        Runs validation and computes the exact MAE on the inspiratory phase.
        """
        self.model.eval()
        total_abs_error = 0.0
        total_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                u_out = x[:, :, 2]
                mask = u_out == 0

                preds = self.model(x).squeeze(-1)

                # Filter for inspiratory phase
                masked_preds = preds[mask]
                masked_targets = y[mask]

                if masked_targets.numel() > 0:
                    # Accumulate sum of absolute errors
                    abs_error = torch.abs(masked_preds - masked_targets).sum()
                    total_abs_error += abs_error.item()
                    total_count += masked_targets.numel()

        # Return global average MAE
        return total_abs_error / total_count if total_count > 0 else 0.0

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Configuration: {Config.EPOCHS} Epochs, Batch Size {Config.BATCH_SIZE}")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_mae = self.validate(val_loader)

            # Step the scheduler based on Validation MAE
            self.scheduler.step(val_mae)

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MAE: {val_mae:.8f} | "
                f"Time: {duration:.2f}s"
            )

            # Save Best Model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def predict(self, test_loader, test_ids):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Loading best model for inference...")
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current weights.")

        self.model.eval()
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for x in test_loader:
                x = x.to(self.device)
                preds = self.model(x).squeeze(-1)
                all_preds.append(preds.cpu().numpy())

        # Concatenate all batches -> Shape: (N_breaths, Seq_Len)
        predictions = np.concatenate(all_preds, axis=0)

        # Flatten to match the submission format (row by row)
        # test_ids is also (N_breaths, Seq_Len), so flattening both aligns them
        flat_preds = predictions.flatten()
        flat_ids = test_ids.flatten()

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": flat_ids, "pressure": flat_preds})

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")


def run_training():
    """
    Entry point to orchestrate the training pipeline.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Initializing Data Pipeline...")
    # prepare_data handles caching and feature engineering automatically
    train_loader, val_loader, test_loader, test_ids = prepare_data()

    # 3. Model Setup
    print("Initializing FMDH-Net Model...")
    model = FMDHNet()

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # 5. Inference
    trainer.predict(test_loader, test_ids)

    print("Run complete.")
