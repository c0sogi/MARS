import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# 1. Patch Configuration for Fast Baseline
import library.config

library.config.NUM_EPOCHS = 5  # Reduce epochs for fast execution
# Ensure directories exist
os.makedirs(library.config.WORKING_DIR, exist_ok=True)
os.makedirs(library.config.SUBMISSION_DIR, exist_ok=True)

# Now import the rest of the library
from library.trainer import train_model
from library.inference import generate_submission
from library.data_loader import process_dataset, GNSSDataset
from library.model import SkyMotionModel
from library.utils import meters_to_degrees_diff, haversine_distance
from library.config import (
    VAL_METADATA_PATH,
    VAL_CACHE_PATH,
    MODEL_CHECKPOINT_PATH,
    DEVICE,
    SKY_FEATURES,
    SCALER_PATH,
    TRAIN_CACHE_PATH,
    TEST_CACHE_PATH,
)
from library.data_loader import CustomScaler


def main():
    print("==================================================")
    print("STARTING PIPELINE")
    print("==================================================")

    # Clean up cache to ensure NaN-free data generation
    print("Cleaning up cache files...")
    for cache_file in [TRAIN_CACHE_PATH, VAL_CACHE_PATH, TEST_CACHE_PATH]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except OSError:
                pass

    # ---------------------------------------------------------
    # 2. Train Model
    # ---------------------------------------------------------
    print("\n[Step 1] Training Model...")
    # load_cached_data=False to force regeneration with new cleaning logic
    trainer = train_model(load_cached_data=False)

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n[Step 2] Evaluating on Validation Set...")

    # Load validation data and metadata explicitly to get phone_names for metric
    val_traj, val_sky, val_y, val_meta = process_dataset(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data=True, is_train=True
    )

    # Load Scaler to transform validation data
    if os.path.exists(SCALER_PATH):
        scaler = CustomScaler()
        scaler.load(SCALER_PATH)
        val_traj, val_sky = scaler.transform(val_traj, val_sky)
    else:
        print("Warning: Scaler not found. Validation data might be unscaled.")

    # Prepare model
    model = SkyMotionModel().to(DEVICE)
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Model checkpoint not found. Using current model state.")
        model = trainer.model

    model.eval()

    # Run Inference in batches to avoid OOM
    batch_size = 1024
    predictions = []

    with torch.no_grad():
        for i in range(0, len(val_traj), batch_size):
            batch_traj = (
                torch.FloatTensor(val_traj[i : i + batch_size])
                .transpose(1, 2)
                .to(DEVICE)
            )
            batch_sky = torch.FloatTensor(val_sky[i : i + batch_size]).to(DEVICE)

            outputs = model(batch_traj, batch_sky)
            predictions.append(outputs.cpu().numpy())

    pred_residuals = np.vstack(predictions)

    # Reconstruct Coordinates
    wls_lat = val_meta["wls_lat"].values
    wls_lon = val_meta["wls_lon"].values

    d_lat_m = pred_residuals[:, 0]
    d_lon_m = pred_residuals[:, 1]

    d_lat_deg, d_lon_deg = meters_to_degrees_diff(d_lat_m, d_lon_m, wls_lat)

    pred_lat = wls_lat + d_lat_deg
    pred_lon = wls_lon + d_lon_deg

    # Calculate Errors

    # Reload original validation metadata to get GT
    df_val_full = pd.read_csv(VAL_METADATA_PATH)
    # Merge on tripId and UnixTimeMillis
    val_eval = pd.merge(
        val_meta,
        df_val_full,
        on=["tripId", "UnixTimeMillis"],
        how="left",
        suffixes=("", "_orig"),
    )

    # Use GT from the merged dataframe
    gt_lat = val_eval["LatitudeDegrees"].values
    gt_lon = val_eval["LongitudeDegrees"].values
    phone_names = val_eval["phone_name"].values

    errors = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    val_eval["error"] = errors

    # Compute Competition Metric
    # Mean of (50th + 95th percentile) averaged over phones
    phones = val_eval["phone_name"].unique()
    phone_scores = []

    print(f"Calculating metric across {len(phones)} phones...")
    for phone in phones:
        p_errors = val_eval[val_eval["phone_name"] == phone]["error"].values
        if len(p_errors) == 0:
            continue
        p50 = np.percentile(p_errors, 50)
        p95 = np.percentile(p_errors, 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    final_metric = np.mean(phone_scores)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 3] Failure Analysis...")

    # Correlate error with Sky Features
    # val_sky is (N, n_features)
    # SKY_FEATURES = ['mean_elev', 'std_elev', 'mean_azim', 'std_azim', 'mean_cn0_sky', 'sat_count']

    print("Correlation between Error Magnitude and Sky Context Features:")
    for i, feature_name in enumerate(SKY_FEATURES):
        feat_values = val_sky[:, i]
        # Handle NaNs just in case
        valid_mask = ~np.isnan(feat_values) & ~np.isnan(errors)
        if np.sum(valid_mask) > 1:
            corr, _ = pearsonr(feat_values[valid_mask], errors[valid_mask])
            print(f"  {feature_name}: {corr:.4f}")
        else:
            print(f"  {feature_name}: N/A (Not enough data)")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    THRESHOLD = 4.256982128481356

    if final_metric < THRESHOLD:
        print(
            f"\n[Step 4] Metric ({final_metric:.4f}) is below threshold ({THRESHOLD:.4f}). Generating Submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\n[Step 4] Metric ({final_metric:.4f}) is NOT below threshold ({THRESHOLD:.4f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
