import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.train import train_model
from library.inference import generate_submission
from library.utils import seed_everything, angular_error
from library.model import DV_AGN
from library.data import IceCubeDataset, process_batch
from library.geometry import load_sensor_geometry


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration for Fast Baseline
    # ---------------------------------------------------------
    seed_everything(Config.SEED)

    print("Initializing Fast Baseline Run...")

    # Create a temporary metadata directory to subset the training data
    # This ensures the training completes quickly within the time limit
    TEMP_META_DIR = os.path.join(Config.WORKING_DIR, "temp_meta")
    os.makedirs(TEMP_META_DIR, exist_ok=True)

    ORIG_META_DIR = Config.METADATA_DIR

    # Load original metadata
    train_meta_path = os.path.join(ORIG_META_DIR, "train_metadata.parquet")
    val_meta_path = os.path.join(ORIG_META_DIR, "val_metadata.parquet")

    train_df = pd.read_parquet(train_meta_path)
    val_df = pd.read_parquet(val_meta_path)

    # Subsample Data: Use 50 batches for training and 5 for internal validation
    # This provides enough diversity for a baseline while keeping runtime low.
    train_batches = train_df["batch_id"].unique()[:50]
    val_batches_internal = val_df["batch_id"].unique()[:5]

    train_subset = train_df[train_df["batch_id"].isin(train_batches)]
    val_subset = val_df[val_df["batch_id"].isin(val_batches_internal)]

    # Save subset metadata
    train_subset.to_parquet(os.path.join(TEMP_META_DIR, "train_metadata.parquet"))
    val_subset.to_parquet(os.path.join(TEMP_META_DIR, "val_metadata.parquet"))

    # Redirect Config to use the subset metadata
    Config.METADATA_DIR = TEMP_META_DIR
    Config.EPOCHS = 2  # Limit epochs for speed

    # ---------------------------------------------------------
    # 2. Model Training
    # ---------------------------------------------------------
    print(
        f"Starting training on {len(train_batches)} batches for {Config.EPOCHS} epochs..."
    )
    train_model(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Full Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nStarting Full Validation...")

    # Restore Config to point to the full metadata for final validation
    Config.METADATA_DIR = ORIG_META_DIR

    # Load the best model saved during training
    device = Config.DEVICE
    model = DV_AGN().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "model.pth")

    if not os.path.exists(model_path):
        print("Error: Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Sensor Geometry
    sensor_map = load_sensor_geometry(Config.SENSOR_GEO_PATH)

    # Get all validation batches
    val_batches_all = val_df["batch_id"].unique()

    all_errors = []

    # Containers for failure analysis
    fa_errors = []
    fa_n_pulses = []
    fa_charge_sum = []
    fa_aux_ratio = []

    MAX_FA_SAMPLES = 100000  # Limit samples for correlation to save memory

    with torch.no_grad():
        for i, batch_id in enumerate(val_batches_all):
            # Process batch
            X_raw, X_canon, targets = process_batch(
                batch_id, val_df, sensor_map, mode="val", load_cached_data=True
            )

            dataset = IceCubeDataset(X_raw, X_canon, targets)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            for xr, xc, y in loader:
                xr = xr.to(device)
                xc = xc.to(device)
                az = y[:, 0].numpy()
                ze = y[:, 1].numpy()

                # Inference
                preds = model(xr, xc)

                # Compute Metric
                batch_errors = angular_error(preds, az, ze)
                all_errors.extend(batch_errors)

                # Collect Failure Analysis Data
                if len(fa_errors) < MAX_FA_SAMPLES:
                    xr_cpu = xr.cpu().numpy()

                    for j in range(len(batch_errors)):
                        if len(fa_errors) >= MAX_FA_SAMPLES:
                            break

                        # Extract features from X_raw: [N, 6] -> x, y, z, t, q, aux
                        evt_data = xr_cpu[j]

                        # Valid pulses have charge > 0 (since log10(0+1)=0 is padding)
                        valid_mask = evt_data[:, 4] > 0

                        if np.any(valid_mask):
                            n_p = np.sum(valid_mask)
                            # Inverse log10 transform for charge: 10^q - 1
                            q_sum = np.sum(10 ** evt_data[valid_mask, 4] - 1)
                            aux_r = np.mean(evt_data[valid_mask, 5])
                        else:
                            n_p = 0
                            q_sum = 0
                            aux_r = 0

                        fa_errors.append(batch_errors[j])
                        fa_n_pulses.append(n_p)
                        fa_charge_sum.append(q_sum)
                        fa_aux_ratio.append(aux_r)

            # Clean up memory
            del X_raw, X_canon, targets, dataset, loader
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute Final Metric
    final_metric = np.mean(all_errors)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Perform Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(fa_errors) > 0:
        # Calculate correlations
        corr_n, _ = pearsonr(fa_errors, fa_n_pulses)
        corr_q, _ = pearsonr(fa_errors, fa_charge_sum)
        corr_a, _ = pearsonr(fa_errors, fa_aux_ratio)

        print(f"Correlation (Error vs N_Pulses): {corr_n:.4f}")
        print(f"Correlation (Error vs Charge_Sum): {corr_q:.4f}")
        print(f"Correlation (Error vs Aux_Ratio): {corr_a:.4f}")

        # Interpretation
        print(
            "Insight: Negative correlation with N_Pulses/Charge implies model struggles with low-energy events."
        )
    else:
        print("Insufficient data for failure analysis.")

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 1.184719

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} meets threshold (< {THRESHOLD}). Generating submission..."
        )
        # Config is already pointing to original metadata (ORIG_META_DIR) which contains test_metadata.parquet
        # But generate_submission uses Config.METADATA_DIR internally.
        # Ensure it points to the correct location.
        Config.METADATA_DIR = ORIG_META_DIR
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric:.6f} does not meet threshold (< {THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
