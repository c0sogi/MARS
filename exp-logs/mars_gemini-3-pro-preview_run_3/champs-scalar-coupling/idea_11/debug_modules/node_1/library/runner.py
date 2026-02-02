import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from library.config import Config
from library.utils import setup_logger, calculate_log_mae, Standardizer
from library.dataset import MolecularGraphDataset
from library.model import ScalarCouplingModel


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the Scalar Coupling Model.
    """

    def __init__(self):
        self.logger = setup_logger("trainer")
        self.device = torch.device(Config.DEVICE)

        # Initialize Standardizer
        self.standardizer = Standardizer(device=self.device)
        # Ensure stats are loaded
        if not self.standardizer.fitted:
            self.standardizer.load()

        # Initialize Model
        self.model = ScalarCouplingModel().to(self.device)

        # Loss Functions
        self.criterion_primary = nn.L1Loss()
        self.criterion_aux = nn.L1Loss()

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

    def get_dataloader(self, split, shuffle=False):
        """Creates a DataLoader for the specified split."""
        dataset = MolecularGraphDataset(split=split, debug=Config.DEBUG)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        total_primary_loss = 0.0
        total_aux_loss = 0.0
        count = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            pred_coupling, pred_shielding, pred_charge = self.model(batch)

            # Calculate Losses
            # Primary: Coupling Constant
            loss_coupling = self.criterion_primary(pred_coupling, batch.y)

            # Auxiliary: Shielding & Charge
            # Filter out NaNs (which might exist if aux data was missing, though handled in prep)
            # In our prep, we filled NaNs with 0 (mean) for test, but for train they should be valid.
            loss_shielding = self.criterion_aux(pred_shielding, batch.aux_shielding)
            loss_charge = self.criterion_aux(pred_charge, batch.aux_charge)

            loss_aux = loss_shielding + loss_charge

            # Composite Loss
            loss = loss_coupling + Config.AUX_LOSS_WEIGHT * loss_aux

            # Backward
            loss.backward()

            # Gradient clipping (optional but good for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            # Track metrics
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            total_primary_loss += loss_coupling.item() * batch_size
            total_aux_loss += loss_aux.item() * batch_size
            count += batch_size

            # Step scheduler if it requires per-step updates (CosineAnnealingWarmRestarts is usually per epoch,
            # but can be per step. Standard usage is per epoch, so we'll step outside).

        avg_loss = total_loss / count
        avg_primary = total_primary_loss / count
        avg_aux = total_aux_loss / count

        return avg_loss, avg_primary, avg_aux

    def validate(self, val_loader):
        """Runs validation and calculates LogMAE."""
        self.model.eval()
        all_preds = []
        all_true = []
        all_types = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward
                pred_coupling, _, _ = self.model(batch)

                # Inverse Transform Predictions
                # pred_coupling is standardized. batch.y is standardized.
                # We need original scale for LogMAE.
                pred_orig = self.standardizer.inverse_transform(
                    pred_coupling, batch.coupling_type
                )
                true_orig = self.standardizer.inverse_transform(
                    batch.y, batch.coupling_type
                )

                all_preds.append(pred_orig.cpu().numpy())
                all_true.append(true_orig.cpu().numpy())
                all_types.append(batch.coupling_type.cpu().numpy())

        # Concatenate
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        types = np.concatenate(all_types)

        # Calculate Metric
        score = calculate_log_mae(y_true, y_pred, types)
        return score

    def fit(self):
        """Main training loop."""
        self.logger.info("Starting training...")

        train_loader = self.get_dataloader("train", shuffle=True)
        val_loader = self.get_dataloader("val", shuffle=False)

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss, train_prim, train_aux = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
                f"Loss: {train_loss:.4f} (Prim: {train_prim:.4f}, Aux: {train_aux:.4f}) | "
                f"Val LogMAE: {val_score:.9f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.logger.info(f"  New best model saved! Score: {best_score:.9f}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        self.logger.info(f"Training complete. Best Val LogMAE: {best_score:.9f}")

    def predict(self):
        """Generates submission file."""
        self.logger.info("Generating predictions for test set...")

        # Load Best Model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError("No model checkpoint found. Train the model first.")

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        test_loader = self.get_dataloader("test", shuffle=False)

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward
                pred_coupling, _, _ = self.model(batch)

                # Inverse Transform
                pred_orig = self.standardizer.inverse_transform(
                    pred_coupling, batch.coupling_type
                )

                ids_list.append(batch.coupling_id.cpu().numpy())
                preds_list.append(pred_orig.cpu().numpy())

        # Aggregate
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)

        # Create DataFrame
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

        # Sort by ID to ensure correct order (though not strictly required by CSV, good practice)
        df_sub.sort_values("id", inplace=True)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Submission shape: {df_sub.shape}")
