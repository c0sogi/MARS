import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    TRAIN_META_PATH,
    WORKING_DIR,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SUBMISSION_DIR,
)
from library.utils import calculate_log_mae, TargetScaler
from library.model_arch import CouplingPredictor


class ModelTrainer:
    """
    Manages the training, validation, and prediction processes for the Scalar Coupling Prediction model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize and fit TargetScaler
        print("Initializing and fitting TargetScaler...")
        self.scaler = TargetScaler()
        train_df = pd.read_csv(TRAIN_META_PATH)
        self.scaler.fit(train_df)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Cosine Annealing Warm Restarts
        # T_0 is the number of iterations for the first restart.
        # We set it to approximate number of batches in an epoch * 5 (restart every 5 epochs initially)
        # or simply set it to T_0=10 epochs.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Loss Function (L1 Loss / MAE)
        self.criterion = nn.L1Loss()

        # State tracking
        self.best_metric = float("inf")
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            # Prepare targets
            targets = batch.y
            types = batch.type_idx.view(-1)

            # Scale targets for training stability
            scaled_targets = self.scaler.transform(targets, types)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(batch)

            # Calculate loss
            # preds shape: (B, 1), scaled_targets shape: (B, 1)
            loss = self.criterion(preds, scaled_targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

        avg_loss = total_loss / count
        return avg_loss

    def validate(self):
        """
        Runs validation and calculates the LogMAE metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_scaled = self.model(batch)

                # Get metadata
                targets = batch.y
                types = batch.type_idx.view(-1)

                # Inverse transform predictions to original scale
                preds_original = self.scaler.inverse_transform(preds_scaled, types)

                all_preds.append(preds_original.cpu())
                all_targets.append(targets.cpu())
                all_types.append(types.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_types = torch.cat(all_types, dim=0)

        # Calculate metric
        metric = calculate_log_mae(all_preds, all_targets, all_types)
        return metric.item()

    def run(self):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metric = self.validate()

            # Scheduler Step
            self.scheduler.step()

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss} - Val LogMAE: {val_metric}"
            )

            # Checkpoint & Early Stopping
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with LogMAE: {val_metric}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best LogMAE: {self.best_metric}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        Returns a DataFrame formatted for submission.
        """
        if self.test_loader is None:
            raise ValueError("Test loader was not provided during initialization.")

        print("Loading best model for inference...")
        if not os.path.exists(self.best_model_path):
            print("Warning: Best model not found, using current model state.")
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()
        ids_list = []
        preds_list = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_scaled = self.model(batch)

                # Inverse transform
                types = batch.type_idx.view(-1)
                preds_original = self.scaler.inverse_transform(preds_scaled, types)

                # Collect IDs and Predictions
                ids_list.append(batch.id.cpu().numpy())
                preds_list.append(preds_original.cpu().numpy())

        # Flatten results
        all_ids = np.concatenate(ids_list).flatten()
        all_preds = np.concatenate(preds_list).flatten()

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id": all_ids, "scalar_coupling_constant": all_preds}
        )

        # Sort by ID to ensure correct order (though usually not strictly required if IDs match)
        submission_df = submission_df.sort_values("id").reset_index(drop=True)

        # Save submission
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

        return submission_df
