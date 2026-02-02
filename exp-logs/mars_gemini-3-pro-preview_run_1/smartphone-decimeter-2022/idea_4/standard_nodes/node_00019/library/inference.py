import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, enu_to_geodetic
from library.data_processing import process_dataset
from library.dataset import GNSSSequenceDataset, collate_padded


def generate_submission(model, feature_stats, load_cached_data=True):
    """
    Generates predictions for the test set using the trained model and saves the submission file.

    Args:
        model (nn.Module): Trained PyTorch model.
        feature_stats (dict): Normalization statistics from training.
        load_cached_data (bool): Whether to load pre-processed test data from cache.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print("Loading test data...")
    # process_dataset handles caching internally for the dataframe
    test_df = process_dataset(
        Config.TEST_METADATA_PATH, load_cached_data=load_cached_data, split_name="test"
    )

    print("Creating test dataset...")
    # GNSSSequenceDataset handles caching internally for the sequences
    test_dataset = GNSSSequenceDataset(
        test_df,
        split_name="test",
        feature_stats=feature_stats,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_padded,
        pin_memory=True,
    )

    model.eval()
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"]  # CPU
            timestamps = batch["timestamps"]  # CPU
            drive_ids = batch["drive_id"]
            phone_names = batch["phone_name"]

            # Forward pass
            # outputs shape: (Batch, Output_Channels, Length)
            outputs = model(features).cpu().numpy()

            # Iterate through batch to extract valid sequence parts
            for i in range(len(drive_ids)):
                length = lengths[i]
                drive_id = drive_ids[i]
                phone_name = phone_names[i]

                # Extract valid sequence parts (remove padding)
                # outputs[i] is (C, L_padded) -> Transpose to (L_padded, C) to iterate over time
                # We slice [:length] to ignore padded timesteps
                pred_seq = outputs[i].T[:length]  # (L, 2)
                time_seq = timestamps[i][:length].numpy()

                for t, (lat_res, lon_res) in enumerate(pred_seq):
                    results.append(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "UnixTimeMillis": time_seq[t],
                            "lat_res_m": lat_res,
                            "lon_res_m": lon_res,
                        }
                    )

    # Create DataFrame from predictions
    pred_df = pd.DataFrame(results)

    # Merge with original test_df to get Baseline coordinates
    # Ensure types match for merge
    pred_df["UnixTimeMillis"] = pred_df["UnixTimeMillis"].astype(np.int64)
    test_df["UnixTimeMillis"] = test_df["UnixTimeMillis"].astype(np.int64)

    # Inner join matches predictions to baselines
    # Note: test_df contains the baseline WLS positions computed during processing
    merged_df = pd.merge(
        test_df, pred_df, on=["drive_id", "phone_name", "UnixTimeMillis"], how="inner"
    )

    # Apply corrections: Baseline + Predicted Residuals (ENU -> Geodetic)
    # The model predicts residuals in meters (North, East)
    pred_lats, pred_lons = enu_to_geodetic(
        merged_df["lat_res_m"].values,
        merged_df["lon_res_m"].values,
        merged_df["BaselineLat"].values,
        merged_df["BaselineLon"].values,
    )

    merged_df["LatitudeDegrees"] = pred_lats
    merged_df["LongitudeDegrees"] = pred_lons

    # Format submission
    # Ensure all required columns are present. tripId is in test_df from metadata.
    # If tripId was lost during processing (it wasn't in GNSS_COLS), we reconstruct it or ensure it's in test_df
    if "tripId" not in merged_df.columns:
        # Reconstruct tripId if missing (drive_id + '-' + phone_name)
        merged_df["tripId"] = merged_df["drive_id"] + "-" + merged_df["phone_name"]

    submission_df = merged_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    # Sort by tripId and timestamp for consistency
    submission_df = submission_df.sort_values(["tripId", "UnixTimeMillis"])

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
