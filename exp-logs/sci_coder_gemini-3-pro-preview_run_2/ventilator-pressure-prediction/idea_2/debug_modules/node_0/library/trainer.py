import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data_loader import get_data_loaders
from library.model import HybridLSTMTransformer


class Trainer:
    """
    Trainer class for the Hybrid LSTM-Transformer model.
    Handles training, validation, checkpointing, and inference.
    """

    def __init__(self, load_cached_data=True, debug=False):
        self.device = torch.device(Config.DEVICE)
        self.debug = debug

        # Load DataLoaders (Data processing and caching handled in data_loader.py)
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            load_cached_data=load_cached_data, debug=debug
        )

        # Initialize Model
        self.model = HybridLSTMTransformer().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        # Using CosineAnnealingWarmRestarts as requested.
        # T_0 is set to total epochs to provide a single cosine decay cycle
        # (effectively CosineAnnealingLR) unless restarts are desired.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Loss Function (MAE)
        self.criterion = nn.L1Loss()

        # Load scaler parameters for metric calculation (unscaling u_out)
        self.u_out_center = 0.0
        self.u_out_scale = 1.0

        if os.path.exists(Config.SCALER_CACHE):
            try:
                scaler_params = np.load(Config.SCALER_CACHE)
                # u_out is at index 2 in CONT_FEATURES
                # CONT_FEATURES = ["time_step", "u_in", "u_out", ...]
                if "u_out" in Config.CONT_FEATURES:
                    u_out_idx = Config.CONT_FEATURES.index("u_out")
                    self.u_out_center = scaler_params["center"][u_out_idx]
                    self.u_out_scale = scaler_params["scale"][u_out_idx]
            except Exception as e:
                print(f"Warning: Could not load scaler params: {e}")

    def train_epoch(self, epoch_idx):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            # Move data to device
            cont_x = batch["cont"].to(self.device)
            cat_x = batch["cat"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(cont_x, cat_x)

            # Compute loss (Full sequence)
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        # Update scheduler
        self.scheduler.step()

        return running_loss / total_samples

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        running_mae_insp = 0.0
        total_samples = 0
        total_insp_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                cont_x = batch["cont"].to(self.device)
                cat_x = batch["cat"].to(self.device)
                targets = batch["target"].to(self.device)

                preds = self.model(cont_x, cat_x)

                # 1. Full Sequence Loss (for monitoring)
                loss = self.criterion(preds, targets)
                running_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

                # 2. Inspiratory Phase MAE (Competition Metric)
                # Recover u_out from scaled features
                u_out_scaled = cont_x[:, :, 2].cpu().numpy()
                u_out = u_out_scaled * self.u_out_scale + self.u_out_center

                # Identify inspiratory phase (u_out == 0)
                # Use 0.5 threshold for binary recovery
                insp_mask = u_out < 0.5

                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Compute MAE only on inspiratory steps
                if insp_mask.sum() > 0:
                    mae_insp = np.abs(preds_np[insp_mask] - targets_np[insp_mask]).sum()
                    running_mae_insp += mae_insp
                    total_insp_samples += insp_mask.sum()

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        avg_mae_insp = (
            running_mae_insp / total_insp_samples if total_insp_samples > 0 else 0.0
        )

        return avg_loss, avg_mae_insp

    def fit(self, epochs=Config.EPOCHS):
        """Main training loop with Early Stopping."""
        print(f"Starting training on device: {self.device}")

        best_val_mae = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mae = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE (Insp): {val_mae}"
            )

            # Early Stopping & Checkpointing
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"Saved best model. New Best MAE: {best_val_mae}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement."
                )
                break

        print(f"Training complete. Best Val MAE: {best_val_mae}")

    def predict(self):
        """Generates predictions for the test set and saves submission file."""
        print("Starting prediction...")

        # Load best model
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading model from {Config.BEST_MODEL_PATH}")
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                cont_x = batch["cont"].to(self.device)
                cat_x = batch["cat"].to(self.device)
                ids = batch["ids"].numpy()  # (Batch, Seq)

                preds = self.model(cont_x, cat_x)  # (Batch, Seq)

                # Flatten sequences
                all_ids.append(ids.flatten())
                all_preds.append(preds.cpu().numpy().flatten())

        # Concatenate all batches
        final_ids = np.concatenate(all_ids)
        final_preds = np.concatenate(all_preds)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": final_ids, "pressure": final_preds})

        # Save to CSV
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
