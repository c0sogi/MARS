import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.trainer import Trainer
from library.dataset import GNSSHeatmapDataset
from library.geo_utils import enu_to_wgs84


def train(load_cached_data=True):
    """
    Trains the Cyclic Spatio-Temporal 2D ResUNet model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
                                 If False, data will be re-processed from raw files.
    """
    print(f"Initializing Training on device: {Config.DEVICE}")

    # 1. Load Datasets
    # The dataset class handles loading from cache or processing from scratch via DataProcessor
    train_dataset = GNSSHeatmapDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = GNSSHeatmapDataset(split="val", load_cached_data=load_cached_data)

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Run Training Loop
    # Trainer handles optimization, loss calculation (including deep supervision),
    # validation, and checkpointing the best model.
    trainer.fit(train_loader, val_loader)


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the trained model.

    This function:
    1. Loads the test dataset (processing it into heatmaps if not cached).
    2. Loads the best trained model checkpoint.
    3. Runs batched inference to predict ENU residuals (East, North).
    4. Aggregates predictions for overlapping windows (Test Time Augmentation).
    5. Converts residuals + WLS baseline back to WGS84 Latitude/Longitude.
    6. Merges predictions with the sample submission file and saves the result.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    print("Initializing Inference...")

    # 1. Load Test Dataset
    test_dataset = GNSSHeatmapDataset(split="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Trainer and Load Best Model
    trainer = Trainer()
    trainer.load_best_model()
    model = trainer.model
    model.eval()

    device = torch.device(Config.DEVICE)

    # Container for aggregated results
    results = []

    print("Running Inference...")
    # We use a custom loop instead of trainer.predict because we need to track
    # metadata (drive_id, phone_name) which is not returned by the standard predict method.
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            features = batch["features"].to(device)
            mask = batch["mask"].cpu().numpy()  # Shape: (B, T)
            wls_pos = batch["wls_pos"].numpy()  # Shape: (B, T, 3)
            timestamps = batch["timestamps"].numpy()  # Shape: (B, T)
            drive_indices = batch["drive_idx"].numpy()  # Shape: (B,)

            # Forward pass
            # Output shape: (Batch, Time, 2) -> (dEast, dNorth)
            residuals = model(features).cpu().numpy()

            batch_size = features.shape[0]
            window_size = features.shape[2]  # Time dimension

            # Unpack batch
            for i in range(batch_size):
                drive_idx = drive_indices[i]
                # Retrieve metadata for this drive
                drive_info = test_dataset.drives[drive_idx]
                drive_id = drive_info["drive_id"]
                phone_name = drive_info["phone_name"]

                # Iterate over time steps in the window
                for t in range(window_size):
                    # Check mask (1.0 = valid data, 0.0 = padding)
                    if mask[i, t] > 0.5:
                        ts = timestamps[i, t]
                        wls = wls_pos[i, t]
                        res = residuals[i, t]

                        results.append(
                            {
                                "drive_id": drive_id,
                                "phone_name": phone_name,
                                "UnixTimeMillis": ts,
                                "dEast": res[0],
                                "dNorth": res[1],
                                "wls_lat": wls[0],
                                "wls_lon": wls[1],
                                "wls_alt": wls[2],
                            }
                        )

    if not results:
        print("Warning: No predictions generated. Check dataset processing.")
        return

    # 3. Aggregate predictions
    # Since windows overlap, we average the residuals for the same timestamp.
    print("Aggregating predictions...")
    df_res = pd.DataFrame(results)

    # Group by unique key and average residuals
    df_agg = (
        df_res.groupby(["drive_id", "phone_name", "UnixTimeMillis"])
        .agg(
            {
                "dEast": "mean",
                "dNorth": "mean",
                "wls_lat": "first",  # WLS baseline is constant for a timestamp
                "wls_lon": "first",
                "wls_alt": "first",
            }
        )
        .reset_index()
    )

    # 4. Apply corrections (ENU -> WGS84)
    print("Applying coordinate corrections...")
    # We assume dUp = 0 for horizontal correction
    pred_lat, pred_lon, _ = enu_to_wgs84(
        df_agg["dEast"].values,
        df_agg["dNorth"].values,
        np.zeros(len(df_agg)),
        df_agg["wls_lat"].values,
        df_agg["wls_lon"].values,
        df_agg["wls_alt"].values,
    )

    df_agg["LatitudeDegrees"] = pred_lat
    df_agg["LongitudeDegrees"] = pred_lon

    # Construct tripId for merging
    df_agg["tripId"] = df_agg["drive_id"] + "-" + df_agg["phone_name"]

    # 5. Merge with Sample Submission
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

    df_sample = pd.read_csv(sample_sub_path)

    # Merge predictions onto sample submission
    # We use a left join on the sample submission to ensure we output exactly the required rows
    df_final = df_sample.merge(
        df_agg[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        how="left",
        suffixes=("", "_pred"),
    )

    # Update values where we have predictions
    mask_pred = ~df_final["LatitudeDegrees_pred"].isna()

    # If prediction exists, use it. Otherwise keep original (WLS baseline from sample_submission)
    df_final.loc[mask_pred, "LatitudeDegrees"] = df_final.loc[
        mask_pred, "LatitudeDegrees_pred"
    ]
    df_final.loc[mask_pred, "LongitudeDegrees"] = df_final.loc[
        mask_pred, "LongitudeDegrees_pred"
    ]

    # Clean up
    df_final = df_final.drop(columns=["LatitudeDegrees_pred", "LongitudeDegrees_pred"])

    # 6. Save Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_final.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(f"Total predictions merged: {mask_pred.sum()} / {len(df_final)}")
