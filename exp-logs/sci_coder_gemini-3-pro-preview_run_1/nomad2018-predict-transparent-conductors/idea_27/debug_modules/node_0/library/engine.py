import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import CRR_DS_Model
from library.data import get_dataloaders
from library.utils import inverse_transform_targets


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the CRR-DS model.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = CRR_DS_Model().to(self.device)

        # Loss function: MSE on log-transformed targets implies optimizing MSLE
        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            atomic_feats = batch["atomic_features"].to(self.device)
            global_feats = batch["global_features"].to(self.device)
            batch_indices = batch["batch_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(atomic_feats, global_feats, batch_indices)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns the average MSE loss (which corresponds to MSLE in original scale).
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_feats = batch["atomic_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(atomic_feats, global_feats, batch_indices)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        train_loader, val_loader, _ = get_dataloaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            duration = time.time() - start_time

            # Print metrics with full precision as requested
            print(f"Epoch {epoch}/{epochs} | Time: {duration:.2f}s")
            print(f"  Train Loss (MSE of log-targets): {train_loss}")
            print(f"  Val Loss (MSE of log-targets): {val_loss}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  -> New best model saved to {Config.BEST_MODEL_PATH}")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def generate_submission(self):
        """
        Loads the best model, predicts on the test set, inverse transforms the results,
        and saves the submission CSV.
        """
        print("Generating submission...")

        # Load test data
        _, _, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=True
        )

        # Load best model weights
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Best model not found at {Config.BEST_MODEL_PATH}. Run fit() first."
            )

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_feats = batch["atomic_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                ids = batch["id"]

                outputs = self.model(atomic_feats, global_feats, batch_indices)

                # Collect IDs and predictions
                all_ids.extend(ids)
                all_preds.append(outputs.cpu().numpy())

        # Concatenate predictions
        all_preds = np.vstack(all_preds)

        # Inverse transform: exp(y) - 1
        # The model predicts log(1 + y), so we need to reverse this to get original energy scale
        original_scale_preds = inverse_transform_targets(all_preds)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {
                "id": all_ids,
                "formation_energy_ev_natom": original_scale_preds[:, 0],
                "bandgap_energy_ev": original_scale_preds[:, 1],
            }
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
