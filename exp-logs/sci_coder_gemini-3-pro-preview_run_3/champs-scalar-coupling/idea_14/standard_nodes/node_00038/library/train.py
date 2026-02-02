import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time

from library.config import Config
from library.utils import AverageMeter, calculate_log_mae, set_seed
from library.data import get_dataloaders
from library.model import SPCFN


class Trainer:
    """
    Manages the training, validation, and inference of the SP-CFN model.
    """

    def __init__(self, load_cached_data=True):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Data Loaders
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader, self.standardizer = (
            get_dataloaders(load_cached_data=load_cached_data)
        )
        # Ensure standardizer stats are on the correct device
        # (The standardizer methods handle tensor creation, but we ensure consistency here)

        # Model
        print("Initializing Model...")
        self.model = SPCFN().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        # Loss Functions
        self.criterion_coupling = nn.L1Loss()
        self.criterion_aux = nn.MSELoss()

        # Training State
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        coupling_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        for i, batch in enumerate(self.train_loader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred_coupling, pred_shielding, pred_charge = self.model(batch)

            # Prepare Targets
            # 1. Standardize Coupling Targets
            target_coupling_raw = batch["y"]
            coupling_types = batch["coupling_type"]
            target_coupling_std = self.standardizer.transform(
                target_coupling_raw, coupling_types
            )

            # 2. Standardize Aux Targets
            target_shielding_raw = batch["aux_s"]
            target_charge_raw = batch["aux_c"]
            target_shielding_std, target_charge_std = self.standardizer.transform_aux(
                target_shielding_raw, target_charge_raw
            )

            # Compute Losses
            loss_c = self.criterion_coupling(pred_coupling, target_coupling_std)

            loss_s = self.criterion_aux(pred_shielding, target_shielding_std)
            loss_ch = self.criterion_aux(pred_charge, target_charge_std)

            loss_aux = loss_s + loss_ch

            # Total Loss
            loss = loss_c + Config.LAMBDA_AUX * loss_aux

            # Backward
            loss.backward()
            self.optimizer.step()

            # Update Scheduler (if strictly per iteration, but usually per epoch for CosineAnnealingWarmRestarts unless specified otherwise.
            # However, PyTorch docs say step() should be called at batch level for CosineAnnealingWarmRestarts?
            # Actually, T_0 is usually epochs. Let's step scheduler at epoch level to be safe/standard,
            # or follow common practice. Given T_0=10, it implies epochs. We'll step at end of epoch.)

            # Metrics
            batch_size = batch["y"].size(0)
            loss_meter.update(loss.item(), batch_size)
            coupling_loss_meter.update(loss_c.item(), batch_size)
            aux_loss_meter.update(loss_aux.item(), batch_size)

        return loss_meter.avg, coupling_loss_meter.avg, aux_loss_meter.avg

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward
                pred_coupling_std, _, _ = self.model(batch)

                # Inverse Transform Predictions
                coupling_types = batch["coupling_type"]
                pred_coupling_raw = self.standardizer.inverse_transform(
                    pred_coupling_std, coupling_types
                )

                # Collect
                all_preds.append(pred_coupling_raw.cpu().numpy())
                all_targets.append(batch["y"].cpu().numpy())
                all_types.append(coupling_types.cpu().numpy())

        # Concatenate
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        types = np.concatenate(all_types)

        # Calculate Metric
        # Map integer types back to string types for the metric function if needed,
        # but calculate_log_mae handles grouping by the passed 'types' array.
        # Ideally, we pass the raw string types or ensure the metric function handles int types.
        # The provided calculate_log_mae groups by 'type' column. Integers are fine for grouping.
        # However, to be strictly correct with competition metric which averages across specific named types,
        # we should ensure the mapping is correct. The metric function just averages whatever groups exist.
        # Since we have a fixed TYPE_MAP, integer grouping is mathematically equivalent to string grouping.

        score = calculate_log_mae(y_true, y_pred, types)
        return score

    def fit(self, max_epochs=Config.MAX_EPOCHS):
        print(f"Starting training for {max_epochs} epochs...")

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_c_loss, train_aux_loss = self.train_epoch(epoch)

            # Step Scheduler
            self.scheduler.step(
                epoch
                + (epoch / len(self.train_loader) if len(self.train_loader) > 0 else 0)
            )

            # Validate
            val_score = self.validate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.6f} (C: {train_c_loss:.6f}, Aux: {train_aux_loss:.6f}) | "
                f"Val LogMAE: {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score < self.best_score:
                print(
                    f"score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"EarlyStopping counter: {self.patience_counter} out of {Config.PATIENCE}"
                )
                if self.patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def predict(self):
        print("Loading best model for inference...")
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()

        ids_list = []
        preds_list = []

        print("Generating predictions on Test set...")
        with torch.no_grad():
            for batch in self.test_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward
                pred_coupling_std, _, _ = self.model(batch)

                # Inverse Transform
                coupling_types = batch["coupling_type"]
                pred_coupling_raw = self.standardizer.inverse_transform(
                    pred_coupling_std, coupling_types
                )

                # Collect
                ids_list.append(batch["id"].cpu().numpy())
                preds_list.append(pred_coupling_raw.cpu().numpy())

        # Concatenate
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)

        # Create DataFrame
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

        # Sort by ID (required by submission format)
        df_sub.sort_values("id", inplace=True)

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def main(load_cached_data=True):
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    trainer = Trainer(load_cached_data=load_cached_data)
    trainer.fit()
    trainer.predict()
