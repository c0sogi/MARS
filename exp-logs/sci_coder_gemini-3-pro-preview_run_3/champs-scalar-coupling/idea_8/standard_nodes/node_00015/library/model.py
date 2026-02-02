import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from library.config import Config
from library.data import get_data_loaders
from library.layers import SDG_CFC
from library.utils import TargetScaler


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the SDG-CFC model.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = SDG_CFC().to(self.device)
        self.scaler = None  # To be initialized with data loaders

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=Config.MIN_LR
        )

        # Loss Function (MSE is appropriate since targets are standardized)
        self.criterion = nn.MSELoss()

    def train_epoch(self, loader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward Pass
            pred_c, pred_s, pred_q = self.model(batch)

            # Standardize Targets
            # Primary target: Coupling Constant
            target_c = self.scaler.transform(batch.y, batch.coupling_type)

            # Auxiliary targets: Shielding and Charges
            target_s, target_q = self.scaler.transform_auxiliary(
                batch.aux_shielding, batch.aux_charge
            )

            # Compute Losses
            loss_c = self.criterion(pred_c, target_c)
            loss_s = self.criterion(pred_s, target_s)
            loss_q = self.criterion(pred_q, target_q)

            # Weighted Sum: Primary + lambda * Auxiliary
            loss = loss_c + Config.AUX_LOSS_WEIGHT * (loss_s + loss_q)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        """
        Evaluates the model on the validation set and computes the competition metric.
        Metric: Mean of Log(MAE) across coupling types.
        """
        self.model.eval()
        type_errors = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward
                pred_c, _, _ = self.model(batch)

                # Inverse Transform to Physical Scale
                pred_real = self.scaler.inverse_transform(pred_c, batch.coupling_type)
                target_real = batch.y

                # Absolute Error
                abs_err = torch.abs(pred_real - target_real)

                # Aggregate by Type
                types = batch.coupling_type.cpu().numpy()
                errors = abs_err.cpu().numpy()

                for t_idx, err in zip(types, errors):
                    type_name = Config.COUPLING_TYPES[t_idx]
                    type_errors[type_name].append(err)

        # Compute Metric: Mean(Log(MAE_type))
        log_maes = []
        print("\nValidation Metrics per Type:")
        for t_name in Config.COUPLING_TYPES:
            if t_name in type_errors and len(type_errors[t_name]) > 0:
                mae = np.mean(type_errors[t_name])
                # Metric is Log of MAE. Adding epsilon only if MAE is exactly 0 (unlikely).
                log_mae = np.log(mae + 1e-9)
                log_maes.append(log_mae)
                print(f"  {t_name}: MAE={mae:.4f}, LogMAE={log_mae:.4f}")
            else:
                print(f"  {t_name}: No samples in validation")

        avg_log_mae = np.mean(log_maes) if log_maes else float("inf")
        return avg_log_mae

    def run(self):
        """
        Main execution loop: Setup -> Train -> Validate -> Predict.
        """
        # Setup
        Config.setup()
        print(f"Device: {self.device}")

        # Data Loading
        train_loader, val_loader, test_loader, scaler = get_data_loaders()
        self.scaler = scaler

        # Training Loop
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

        for epoch in range(Config.MAX_EPOCHS):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (Mean LogMAE): {val_score:.6f} | "
                f"LR: {current_lr:.2e}"
            )

            # Checkpointing & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved! Score: {best_score:.6f}")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Inference
        self.predict_and_submit(test_loader)

    def predict_and_submit(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("\nGenerating submission...")
        # Load Best Model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model weights.")
        else:
            print("Warning: No model weights found. Using current weights.")

        self.model.eval()
        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward
                pred_c, _, _ = self.model(batch)

                # Inverse Transform
                pred_real = self.scaler.inverse_transform(pred_c, batch.coupling_type)

                # Collect
                ids_list.extend(batch.coupling_id.cpu().numpy())
                preds_list.extend(pred_real.cpu().numpy())

        # Save
        df_sub = pd.DataFrame({"id": ids_list, "scalar_coupling_constant": preds_list})

        # Ensure ID is int
        df_sub["id"] = df_sub["id"].astype(int)

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions: {len(df_sub)}")


def main():
    trainer = Trainer()
    trainer.run()
