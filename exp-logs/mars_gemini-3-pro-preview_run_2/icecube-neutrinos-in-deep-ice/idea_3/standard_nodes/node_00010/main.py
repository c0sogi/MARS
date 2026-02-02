import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_META_PATH,
    seed_everything,
)
from library.trainer import train_model
from library.inference import predict_and_submit
from library.data_loader import get_dataloader
from library.model import TemporalCNN
from library.utils import direction_to_angles, angular_dist_score


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Initializing Fast Baseline Run...")

    # Define constraints for fast execution
    # 500k samples is approx 0.5% of the dataset, sufficient for a baseline check
    # 5 epochs allows for convergence on this subset without exceeding time limits
    TRAIN_SAMPLES = 500_000
    VAL_SAMPLES = 50_000
    RUN_EPOCHS = 5

    # 2. Training Phase
    print(
        f"\n[Step 1/3] Training Model (Max Samples: {TRAIN_SAMPLES}, Epochs: {RUN_EPOCHS})..."
    )
    best_model_path = train_model(
        max_train_samples=TRAIN_SAMPLES, max_val_samples=VAL_SAMPLES, epochs=RUN_EPOCHS
    )

    # 3. Validation & Failure Analysis Phase
    print("\n[Step 2/3] Performing Validation and Failure Analysis...")

    # Load the best model
    model = TemporalCNN().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Create a dataloader specifically for analysis
    val_loader = get_dataloader(
        VAL_META_PATH,
        mode="val",
        max_samples=VAL_SAMPLES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Containers for analysis
    all_errors = []
    feature_stats = {"log_total_charge": [], "num_pulses": [], "aux_ratio": []}

    all_preds_az = []
    all_preds_zen = []
    all_true_az = []
    all_true_zen = []

    # Inference loop on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            preds = model(inputs)
            preds_norm = F.normalize(preds, p=2, dim=1)

            # --- Metric Calculation Prep ---
            # Convert to angles
            p_az, p_zen = direction_to_angles(
                preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
            )
            t_az, t_zen = direction_to_angles(
                targets[:, 0], targets[:, 1], targets[:, 2]
            )

            all_preds_az.extend(p_az.cpu().numpy())
            all_preds_zen.extend(p_zen.cpu().numpy())
            all_true_az.extend(t_az.cpu().numpy())
            all_true_zen.extend(t_zen.cpu().numpy())

            # --- Failure Analysis Prep ---
            # Calculate angular error per sample for correlation
            # Dot product clamped to [-1, 1]
            dot_prod = torch.sum(preds_norm * targets, dim=1)
            dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
            batch_errors = torch.acos(dot_prod).cpu().numpy()
            all_errors.extend(batch_errors)

            # Extract Features from Input Tensor (B, 6, 128)
            # Channel 4: Charge (log1p transformed)
            # Channel 5: Auxiliary (0 or 1)

            # 1. Total Charge (proxy for energy)
            charge_channel = inputs[:, 4, :]
            # Summing log-charge isn't exactly physical total charge, but correlates strongly
            batch_total_charge = torch.sum(charge_channel, dim=1).cpu().numpy()
            feature_stats["log_total_charge"].extend(batch_total_charge)

            # 2. Number of Pulses (count non-zero entries in charge channel)
            # Since padding is 0, and log1p(charge) > 0 for charge > 0
            batch_n_pulses = torch.sum(charge_channel > 0, dim=1).cpu().numpy()
            feature_stats["num_pulses"].extend(batch_n_pulses)

            # 3. Auxiliary Ratio
            aux_channel = inputs[:, 5, :]
            batch_aux_count = torch.sum(aux_channel, dim=1).cpu().numpy()
            # Avoid division by zero
            safe_n_pulses = batch_n_pulses.copy()
            safe_n_pulses[safe_n_pulses == 0] = 1
            batch_aux_ratio = batch_aux_count / safe_n_pulses
            feature_stats["aux_ratio"].extend(batch_aux_ratio)

    # --- Compute Final Metric ---
    y_pred = np.stack([all_preds_az, all_preds_zen], axis=1)
    y_true = np.stack([all_true_az, all_true_zen], axis=1)
    final_metric = angular_dist_score(y_true, y_pred)

    print(f"Final Validation Metric: {final_metric}")

    # --- Compute Failure Correlations ---
    print("\nFailure Analysis (Pearson Correlation with Error Magnitude):")
    errors_arr = np.array(all_errors)

    for name, values in feature_stats.items():
        vals_arr = np.array(values)
        if len(vals_arr) > 1 and np.std(vals_arr) > 0 and np.std(errors_arr) > 0:
            corr, _ = pearsonr(vals_arr, errors_arr)
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (Insufficient variance)")

    # 4. Submission Phase
    print("\n[Step 3/3] Generating Submission for Test Set...")
    # Using the library function which handles the full test set
    predict_and_submit(model_path=best_model_path)

    print("\nRun Complete.")


if __name__ == "__main__":
    main()
