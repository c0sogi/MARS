import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.data_processor import load_data
from library.model import RelativeTrajectoryCNN, GNSSWindowDataset
from library.trainer import run_training, generate_submission
from library.utils import enu_to_ecef, ecef_to_lla, haversine_distance


def evaluate_validation(model, device):
    """
    Performs inference on the validation set, reconstructs coordinates,
    calculates the competition metric, and performs failure analysis.
    """
    print("\n--- Starting Validation Assessment ---")

    # 1. Load Validation Data
    # load_data returns (X, y, meta_df)
    # meta_df contains the WLS baselines and Ground Truth needed for metric calc
    print("Loading validation data...")
    X_val, y_val, df_val_meta = load_data(mode="val", load_cached_data=True)

    # Create DataLoader
    val_dataset = GNSSWindowDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 2. Inference
    model.eval()
    preds_list = []
    targets_list = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())

    preds = np.vstack(preds_list)  # Predicted ENU offsets (East, North)

    # 3. Reconstruct Absolute Coordinates
    # The model predicts the offset from the WLS baseline in meters (East, North)
    # We need to convert these offsets back to Lat/Lon

    print("Reconstructing coordinates...")
    # Extract baselines
    wls_lat = df_val_meta["WlsLat"].values
    wls_lon = df_val_meta["WlsLon"].values

    # Predicted local offsets (DeltaEast, DeltaNorth)
    pred_e = preds[:, 0]
    pred_n = preds[:, 1]

    deg_to_m = 111320.0

    # Inverse scaling
    pred_lat = wls_lat + (pred_n / deg_to_m)

    cos_lat = np.cos(np.radians(wls_lat))
    pred_lon = wls_lon + (pred_e / (deg_to_m * cos_lat))

    # 4. Calculate Distance Errors
    gt_lat = df_val_meta["LatitudeDegrees"].values
    gt_lon = df_val_meta["LongitudeDegrees"].values

    errors = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    df_val_meta["Error"] = errors

    # 5. Compute Competition Metric
    # Mean of (50th + 95th percentile) per phone
    print("Computing competition metric...")

    # Group by tripId (which represents a phone-drive collection)
    trip_metrics = []
    for trip_id, group in df_val_meta.groupby("tripId"):
        errs = group["Error"].values
        p50 = np.percentile(errs, 50)
        p95 = np.percentile(errs, 95)
        trip_metrics.append((p50 + p95) / 2)

    final_metric = np.mean(trip_metrics)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Correlate Error with signal features
    # Features in df_val_meta (from process_features): MeanCn0, MeanUncertainty, SatelliteCount
    features_to_analyze = ["MeanCn0", "MeanUncertainty", "SatelliteCount"]

    print("Correlation between Error and Input Features:")
    for feat in features_to_analyze:
        if feat in df_val_meta.columns:
            # Drop NaNs for correlation
            valid_mask = df_val_meta[feat].notna() & df_val_meta["Error"].notna()
            if valid_mask.sum() > 0:
                corr, _ = pearsonr(
                    df_val_meta.loc[valid_mask, feat],
                    df_val_meta.loc[valid_mask, "Error"],
                )
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found in metadata")

    return final_metric


def main():
    # 1. Train the model
    # We limit epochs to ensure the script completes quickly within the time limit
    print("--- Starting Training Pipeline ---")
    run_training(epochs=5, load_cached_data=True)

    # 2. Load the trained model
    device = Config.DEVICE
    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Error: Model file not found after training.")
        return

    # 3. Evaluate on Validation Set
    val_score = evaluate_validation(model, device)

    # 4. Generate Submission if threshold met
    # Threshold: 4.256982128481356
    THRESHOLD = 4.256982128481356

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation score ({val_score:.6f}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
