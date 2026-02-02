import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.train import run_training
from library.inference import predict_and_submit
from library.model import PointNetBaseline
from library.data import IceCubeBatchDataset
from library.utils import load_sensor_geometry, angles_to_direction, direction_to_angles


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(model, val_meta, sensor_geo, device):
    """
    Runs inference on the validation set, calculates the final metric,
    and performs failure analysis by correlating errors with input features.
    """
    print("Running Validation & Failure Analysis...")

    # Apply the same subsetting logic as training if DEBUG is enabled
    if Config.DEBUG:
        batch_ids = val_meta["batch_id"].unique()
        if len(batch_ids) > 0:
            # Restrict to the first batch and slice
            val_meta = val_meta[val_meta["batch_id"] == batch_ids[0]]
            val_meta = val_meta.iloc[: Config.DEBUG_SUBSET_SIZE]

    batch_ids = val_meta["batch_id"].unique()

    all_errors = []
    all_n_pulses = []
    all_total_charge = []

    model.eval()

    # Disable gradients for inference
    with torch.no_grad():
        for batch_id in batch_ids:
            try:
                # Load validation batch
                dataset = IceCubeBatchDataset(
                    batch_id=batch_id,
                    meta_df=val_meta,
                    sensor_geo=sensor_geo,
                    mode="train",  # mode='train' ensures targets (y) are loaded
                    load_cached_data=True,
                )
            except Exception as e:
                print(f"Skipping validation batch {batch_id}: {e}")
                continue

            if len(dataset) == 0:
                continue

            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            for X, y in loader:
                X = X.to(device)
                y_true = y.numpy()  # (Azimuth, Zenith)

                # Forward pass
                preds = model(X)
                preds = F.normalize(preds, p=2, dim=1)

                # Convert predictions to angles
                az_pred, zen_pred = direction_to_angles(preds)
                y_pred_angles = np.stack(
                    [az_pred.cpu().numpy(), zen_pred.cpu().numpy()], axis=1
                )

                # --- Metric Calculation (Element-wise) ---
                # Convert angles to 3D unit vectors
                vec_true = angles_to_direction(y_true[:, 0], y_true[:, 1])
                vec_pred = angles_to_direction(y_pred_angles[:, 0], y_pred_angles[:, 1])

                # Compute angular error: arccos(dot_product)
                dot = np.sum(vec_true * vec_pred, axis=1)
                dot = np.clip(dot, -1.0, 1.0)  # Clip for numerical stability
                errors = np.arccos(dot)

                all_errors.append(errors)

                # --- Feature Extraction for Analysis ---
                # X shape: (Batch, Pulses, Features)
                # Features: [x, y, z, time, charge, auxiliary]
                X_np = X.cpu().numpy()

                # 1. Pulse Count: Count non-padded rows.
                # Padded rows are all zeros. Check if sum of abs(features) > epsilon
                is_pulse = np.sum(np.abs(X_np), axis=2) > 1e-6
                n_pulses = np.sum(is_pulse, axis=1)
                all_n_pulses.append(n_pulses)

                # 2. Total Charge: Sum of 10^charge for valid pulses
                # Feature index 4 is log10(charge)
                log_charge = X_np[:, :, 4]
                raw_charge = np.power(10, log_charge)
                # Zero out padded pulses (where 10^0 would be 1)
                raw_charge[~is_pulse] = 0
                total_charge = np.sum(raw_charge, axis=1)
                all_total_charge.append(total_charge)

    if not all_errors:
        print("No validation data processed.")
        return

    # Concatenate results
    all_errors = np.concatenate(all_errors)
    all_n_pulses = np.concatenate(all_n_pulses)
    all_total_charge = np.concatenate(all_total_charge)

    # --- 1. Print Final Metric ---
    mean_mae = np.mean(all_errors)
    print(f"Final Validation Metric: {mean_mae:.10f}")

    # --- 2. Failure Analysis ---
    df_analysis = pd.DataFrame(
        {
            "error": all_errors,
            "n_pulses": all_n_pulses,
            "total_charge": all_total_charge,
        }
    )

    corr_pulses = df_analysis["error"].corr(df_analysis["n_pulses"])
    corr_charge = df_analysis["error"].corr(df_analysis["total_charge"])

    print("\n--- Failure Analysis ---")
    print(f"Correlation (MAE vs Pulse Count): {corr_pulses:.4f}")
    print(f"Correlation (MAE vs Total Charge): {corr_charge:.4f}")


def main():
    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 1024
    Config.DEBUG = False
    # Config.DEBUG_SUBSET_SIZE = 50000  # Train on 50k events for speed

    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Training Phase
    # ---------------------------------------------------------
    print("=== Starting Training Phase ===")
    run_training(Config)

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis Phase
    # ---------------------------------------------------------
    print("\n=== Starting Validation & Failure Analysis Phase ===")

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = PointNetBaseline(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    # Load Best Weights
    if os.path.exists(Config.MODEL_PATH):
        try:
            model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=device, weights_only=True)
            )
        except TypeError:
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Error: Model weights not found. Cannot proceed with validation.")
        return

    # Load Metadata and Geometry
    val_meta = pd.read_parquet(Config.VAL_META)
    sensor_geo = load_sensor_geometry(Config.SENSOR_GEOMETRY_PATH)

    # Execute Analysis
    perform_failure_analysis(model, val_meta, sensor_geo, device)

    # Clean up memory before inference
    del val_meta, sensor_geo, model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. Submission Phase
    # ---------------------------------------------------------
    print("\n=== Starting Submission Phase ===")

    # Disable DEBUG to process the FULL test set
    Config.DEBUG = False

    predict_and_submit(Config)


if __name__ == "__main__":
    main()
