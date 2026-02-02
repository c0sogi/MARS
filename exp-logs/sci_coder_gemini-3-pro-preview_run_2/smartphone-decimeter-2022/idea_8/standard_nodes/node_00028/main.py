import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import BiGRUModel
from library.trainer import Trainer
from library.inference import predict_and_submit
from library.preprocessor import process_trip


def reconstruct_validation_metadata():
    """
    Reconstructs the validation metadata DataFrame to align with the preprocessed
    validation data (val_X, val_y). This is necessary to map predictions back
    to specific phones for metric calculation.
    """
    print("Reconstructing validation metadata for metric calculation...")

    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA_PATH}"
        )

    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)
    meta_list = []

    # The preprocessor iterates over groups sorted by tripId (pandas groupby default)
    # We must replicate this logic exactly to ensure alignment.
    for trip_id, group in val_meta_df.groupby("tripId"):
        first = group.iloc[0]
        drive_id = first["drive_id"]
        phone_name = first["phone_name"]
        gnss_path = first["gnss_path"]

        target_ts = group["UnixTimeMillis"].values

        # We only need valid_ts to filter the metadata
        _, valid_ts, _ = process_trip(
            trip_id, drive_id, phone_name, gnss_path, target_timestamps=target_ts
        )

        if valid_ts is None or len(valid_ts) == 0:
            continue

        # Filter metadata to match the valid timestamps processed by the model
        valid_group = group[group["UnixTimeMillis"].isin(valid_ts)].copy()

        # Ensure order matches the processed data (sorted by time)
        valid_group = (
            valid_group.set_index("UnixTimeMillis").reindex(valid_ts).reset_index()
        )
        meta_list.append(valid_group)

    if not meta_list:
        raise RuntimeError("No valid validation data found during reconstruction.")

    return pd.concat(meta_list, ignore_index=True)


def calculate_competition_metric(errors, phone_names):
    """
    Calculates the competition metric:
    Mean of ( (50th percentile + 95th percentile) / 2 ) across all phones.
    """
    df = pd.DataFrame({"error": errors, "phone": phone_names})

    phone_scores = []
    for phone, group in df.groupby("phone"):
        p50 = np.percentile(group["error"], 50)
        p95 = np.percentile(group["error"], 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    return np.mean(phone_scores)


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing...")

    # Override Config for fast baseline execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 1024  # Increase batch size for A100

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # load_cached_data=True will use existing files in ./working/idea_8 if available
    train_loader, val_loader, test_loader, test_meta = get_dataloaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting training...")
    trainer.fit(train_loader, val_loader)

    # -------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing validation inference...")
    device = torch.device(Config.DEVICE)
    model = trainer.model
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Euclidean distance error (Residual Prediction Error)
    # Target y is (dLat_m, dLon_m) relative to WLS. Prediction is the same.
    # Error vector = Pred - Target
    error_vectors = val_preds - val_targets
    distance_errors = np.sqrt(np.sum(error_vectors**2, axis=1))

    # Reconstruct metadata to get phone names
    val_meta_df = reconstruct_validation_metadata()

    # Integrity check
    if len(val_meta_df) != len(distance_errors):
        print(
            f"Warning: Metadata length ({len(val_meta_df)}) does not match prediction length ({len(distance_errors)})."
        )
        # Truncate to the shorter length to allow execution to proceed, though this indicates an alignment issue
        min_len = min(len(val_meta_df), len(distance_errors))
        val_meta_df = val_meta_df.iloc[:min_len]
        distance_errors = distance_errors[:min_len]

    phone_names = val_meta_df["phone_name"].values

    # Compute Metric
    final_metric = calculate_competition_metric(distance_errors, phone_names)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nFailure Analysis:")
    # Load validation features (X) to correlate with errors
    # We need to reload val_X from cache or extract from loader (loader shuffles=False)
    val_X = np.load(Config.CACHE_VAL_X)

    # Features: ['rel_lat_m', 'rel_lon_m', 'vel_lat_m', 'vel_lon_m', 'vel_alt_m', 'raw_pr_unc', 'cn0', 'sat_count']
    # We take the mean over the window for correlation analysis
    # val_X shape: (N, Window, Features)
    val_X_mean = np.mean(val_X, axis=1)

    feature_names = Config.FEATURE_COLUMNS
    correlations = {}

    print("Correlation between Error Magnitude and Input Features:")
    for i, feature in enumerate(feature_names):
        # Handle potential length mismatch if truncation occurred above
        feat_values = val_X_mean[: len(distance_errors), i]
        corr, _ = spearmanr(feat_values, distance_errors)
        correlations[feature] = corr
        print(f"  {feature}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 4.256982128481356

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.4f}) is better than threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True, batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"\nValidation metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    run()
