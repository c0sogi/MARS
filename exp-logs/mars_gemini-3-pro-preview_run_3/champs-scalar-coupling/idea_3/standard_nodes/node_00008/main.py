import sys
import os
import torch
import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.train import Trainer
from library.utils import calculate_log_mae

# ==========================================
# 1. Configuration Overrides for Fast Baseline
# ==========================================
# Modify Config to fit within the 2-hour limit
Config.MAX_EPOCHS = 5
Config.BATCH_SIZE = 128
Config.PATIENCE = 3
Config.NUM_WORKERS = 4
STEPS_PER_EPOCH = 3000  # Limit steps to ensure quick epochs


# ==========================================
# 2. Custom Trainer for Fast Execution
# ==========================================
class FastTrainer(Trainer):
    """
    Subclass of Trainer that limits the number of steps per epoch
    to ensure the baseline runs within the time constraints.
    """

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        count = 0
        steps = 0

        for batch in self.train_loader:
            # Move batch to device
            batch_x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_attr = batch["edge_attr"].to(self.device)
            triplets = batch["triplets"].to(self.device)
            triplet_attr = batch["triplet_attr"].to(self.device)

            coupling_index = batch["coupling_index"].to(self.device)
            coupling_type = batch["coupling_type"].to(self.device)
            coupling_value = batch["coupling_value"].to(self.device)

            aux_targets = batch["aux"].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            pred_coupling, pred_shield, pred_charge = self.model(
                x=batch_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                triplets=triplets,
                triplet_attr=triplet_attr,
                coupling_index=coupling_index,
                coupling_type=coupling_type,
            )

            # --- Loss Calculation ---
            # 1. Main Task: Standardized L1 Loss
            target_std = self.standardizer.transform(coupling_value, coupling_type)
            loss_coupling = torch.nn.functional.l1_loss(pred_coupling, target_std)

            # 2. Auxiliary Tasks
            loss_shield = torch.nn.functional.l1_loss(pred_shield, aux_targets[:, :9])
            loss_charge = torch.nn.functional.l1_loss(
                pred_charge, aux_targets[:, 9].unsqueeze(-1)
            )

            # Weighted Sum
            loss = (
                loss_coupling
                + Config.LAMBDA_SHIELDING * loss_shield
                + Config.LAMBDA_CHARGE * loss_charge
            )

            # Backward
            loss.backward()
            self.optimizer.step()

            # Logging
            bs = coupling_value.size(0)
            total_loss += loss.item() * bs
            total_main_loss += loss_coupling.item() * bs
            total_aux_loss += (loss_shield.item() + loss_charge.item()) * bs
            count += bs
            steps += 1

            if steps >= STEPS_PER_EPOCH:
                break

        avg_loss = total_loss / count if count > 0 else 0.0
        avg_main = total_main_loss / count if count > 0 else 0.0
        avg_aux = total_aux_loss / count if count > 0 else 0.0

        return avg_loss, avg_main, avg_aux

    def run_failure_analysis(self):
        """
        Performs failure analysis on the validation set.
        Calculates correlation between error and target magnitude.
        """
        print("\n--- Failure Analysis ---")
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        # Collect predictions
        with torch.no_grad():
            for batch in self.val_loader:
                batch_x = batch["x"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                edge_attr = batch["edge_attr"].to(self.device)
                triplets = batch["triplets"].to(self.device)
                triplet_attr = batch["triplet_attr"].to(self.device)

                coupling_index = batch["coupling_index"].to(self.device)
                coupling_type = batch["coupling_type"].to(self.device)
                coupling_value = batch["coupling_value"].to(self.device)

                pred_std, _, _ = self.model(
                    x=batch_x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    triplets=triplets,
                    triplet_attr=triplet_attr,
                    coupling_index=coupling_index,
                    coupling_type=coupling_type,
                )

                pred_orig = self.standardizer.inverse_transform(pred_std, coupling_type)

                all_preds.append(pred_orig.cpu().numpy())
                all_targets.append(coupling_value.cpu().numpy())
                all_types.append(coupling_type.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        types = np.concatenate(all_types)

        # Calculate Absolute Errors
        abs_errors = np.abs(y_true - y_pred)
        target_mags = np.abs(y_true)

        # Correlation Analysis
        corr, _ = pearsonr(abs_errors, target_mags)
        print(f"Correlation between Absolute Error and Target Magnitude: {corr:.4f}")

        # Analysis by Type
        print("Mean Absolute Error by Coupling Type:")
        unique_types = np.unique(types)
        for t in unique_types:
            mask = types == t
            type_mae = np.mean(abs_errors[mask])
            type_name = Config.COUPLING_TYPES[int(t)]
            print(f"  {type_name}: {type_mae:.4f}")


# ==========================================
# 3. Main Execution Flow
# ==========================================
def main():
    print("Starting Fast Baseline Run...")
    start_time = time.time()

    # Initialize Trainer
    trainer = FastTrainer()

    # Run Training
    # This will train for Config.MAX_EPOCHS, limiting steps per epoch
    trainer.run_training()

    # Load Best Model for Validation
    print("\nLoading best model for final validation...")
    trainer.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
    )

    # Calculate Final Metric
    final_metric = trainer.validate()
    print(f"Final Validation Metric: {final_metric}")

    # Run Failure Analysis
    trainer.run_failure_analysis()

    # Submission Decision
    # Threshold: -1.2761284112930298
    # LogMAE is better when lower.
    threshold = -1.2761284112930298

    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold})."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )

    elapsed = time.time() - start_time
    print(f"Total Runtime: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()
