import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os

from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_PATH,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.model import DSGCN
from library.data import get_loaders


class Trainer:
    """
    Trainer class for the Dual-Stream Geometric-Compositional Network.
    Handles training, evaluation, checkpointing, and submission generation.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Initialize Model
        self.model = DSGCN().to(self.device)

        # Initialize Optimizer
        # Using AdamW as specified in the idea description
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Loss Function
        self.criterion = nn.MSELoss()

        # Load Data
        # load_cached_data=True ensures we use the preprocessed .npz files if available
        self.train_loader, self.val_loader, self.test_loader, self.scalers = (
            get_loaders(load_cached_data=True)
        )

        # Prepare Scalers for Inverse Transformation
        # Scalers are numpy arrays in the dictionary, convert to Tensor on device
        self.target_mean = torch.tensor(
            self.scalers["target_mean"], device=self.device
        ).float()
        self.target_std = torch.tensor(
            self.scalers["target_std"], device=self.device
        ).float()

    def inverse_transform(self, y_norm):
        """
        Reverts the Z-score normalization applied to targets.
        y = y_norm * std + mean
        """
        return y_norm * self.target_std + self.target_mean

    def compute_rmsle(self, y_pred, y_true):
        """
        Computes Column-wise Root Mean Squared Logarithmic Error.
        Metric = mean( sqrt( mean( (log(p+1) - log(t+1))^2 ) ) ) over columns
        """
        # Ensure non-negative values for log
        y_pred = torch.clamp(y_pred, min=0.0)
        y_true = torch.clamp(y_true, min=0.0)

        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)

        squared_log_error = (log_pred - log_true) ** 2

        # Mean squared log error per column
        msle_per_col = torch.mean(squared_log_error, dim=0)

        # Root mean squared log error per column
        rmsle_per_col = torch.sqrt(msle_per_col)

        # Average over columns (formation energy and bandgap)
        return torch.mean(rmsle_per_col).item()

    def train_one_epoch(self):
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds_norm = self.model(batch)

            # Compute loss on normalized targets
            loss = self.criterion(preds_norm, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

        return total_loss / num_samples

    def evaluate(self, loader):
        """
        Evaluates the model on a given loader.
        Returns average MSE loss (normalized) and RMSLE (original scale).
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_norm = self.model(batch)

                # Loss on normalized data
                loss = self.criterion(preds_norm, batch.y)
                total_loss += loss.item() * batch.num_graphs
                num_samples += batch.num_graphs

                # Inverse transform for metric calculation
                preds = self.inverse_transform(preds_norm)
                targets = self.inverse_transform(batch.y)

                all_preds.append(preds)
                all_targets.append(targets)

        avg_loss = total_loss / num_samples

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Metric
        rmsle = self.compute_rmsle(all_preds, all_targets)

        return avg_loss, rmsle

    def run(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")
        best_val_rmsle = float("inf")
        patience_counter = 0
        start_time = time.time()

        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_rmsle = self.evaluate(self.val_loader)

            # Print metrics
            # Requirement: print full precision for validation metric
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val RMSLE: {val_rmsle}"
            )

            # Early Stopping and Checkpointing
            if val_rmsle < best_val_rmsle:
                best_val_rmsle = val_rmsle
                patience_counter = 0
                torch.save(self.model.state_dict(), CHECKPOINT_PATH)
                print(f"  New best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        print(f"Best Validation RMSLE: {best_val_rmsle}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        print("Generating submission...")

        # Load best model weights
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint file not found at {CHECKPOINT_PATH}. Train the model first."
            )

        self.model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)

                # Predict
                preds_norm = self.model(batch)

                # Inverse transform
                preds = self.inverse_transform(preds_norm)

                # Clip negative values to 0 (physical constraint)
                preds = torch.clamp(preds, min=0.0)

                # Convert to numpy
                ids = batch.id.cpu().numpy()
                preds_np = preds.cpu().numpy()

                # Collect results
                for i, sample_id in enumerate(ids):
                    results.append(
                        {
                            "id": int(sample_id),
                            "formation_energy_ev_natom": preds_np[i, 0],
                            "bandgap_energy_ev": preds_np[i, 1],
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(results)

        # Sort by ID just in case
        df = df.sort_values("id")

        # Ensure correct column order
        cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        df = df[cols]

        # Save to CSV
        df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
