import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    CACHE_DIR,
    SUBMISSION_DIR,
    SAMPLE_SUBMISSION_PATH,
)
from library.model import RelativeWindowedMLP
from library.data_loader import load_dataset
from library.utils import meters_to_wgs84_relative


def generate_predictions(scaler=None, load_cached=True):
    """
    Generates predictions for the test set using the trained model and saves the submission file.

    Args:
        scaler: Optional sklearn scaler. If None, it will be loaded from the cache directory
                by the data loader.
        load_cached (bool): Whether to load pre-processed data from cache if available.
    """
    print("Initializing Inference Pipeline...")

    # 1. Load Test Data
    # load_dataset handles loading the scaler from disk if it's not provided
    print("Loading test dataset...")
    test_dataset, _ = load_dataset(
        mode="test", scaler=scaler, load_cached_data=load_cached
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 2. Load Model
    print("Loading trained model...")
    model = RelativeWindowedMLP().to(DEVICE)
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {best_model_path}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Run Inference
    print("Running inference on test data...")
    preds_residuals = []

    with torch.no_grad():
        for batch in test_loader:
            traj = batch["traj_feat"].to(DEVICE)
            sky = batch["sky_feat"].to(DEVICE)

            # Model outputs [delta_x (East), delta_y (North)] in meters
            outputs = model(traj, sky)
            preds_residuals.append(outputs.cpu().numpy())

    # Concatenate all batch predictions
    if len(preds_residuals) > 0:
        pred_residuals_np = np.concatenate(preds_residuals, axis=0)
    else:
        pred_residuals_np = np.empty((0, 2))

    # 4. Reconstruct Absolute Coordinates
    # The dataset metadata contains [trip_id, timestamp, wls_lat, wls_lon]
    # We add the predicted meter offsets to the WLS baseline
    test_meta = test_dataset.meta

    pred_lats = []
    pred_lons = []

    print(f"Reconstructing coordinates for {len(test_meta)} samples...")
    for i in range(len(test_meta)):
        # Extract baseline WLS position from metadata
        wls_lat = test_meta[i, 2]
        wls_lon = test_meta[i, 3]

        # Predicted residuals
        dx = pred_residuals_np[i, 0]  # Easting offset in meters
        dy = pred_residuals_np[i, 1]  # Northing offset in meters

        # Convert metric offset back to degrees
        lat, lon = meters_to_wgs84_relative(wls_lat, wls_lon, dx, dy)

        pred_lats.append(lat)
        pred_lons.append(lon)

    # 5. Format Submission
    print("Formatting submission...")
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta[:, 0],
            "UnixTimeMillis": test_meta[:, 1],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    # Ensure UnixTimeMillis is integer for correct merging
    submission_df["UnixTimeMillis"] = submission_df["UnixTimeMillis"].astype(np.int64)

    # Load sample submission to ensure correct row order and completeness
    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission file not found at {SAMPLE_SUBMISSION_PATH}"
        )

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Merge predictions into sample submission structure
    # We use a left join on the sample submission to ensure we have all required rows
    final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
        submission_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Fill missing predictions with values from sample submission (usually WLS/Baseline)
    # This handles cases where data might have been filtered out or missing in preprocessing
    missing_mask = final_sub["LatitudeDegrees"].isna()
    if missing_mask.sum() > 0:
        print(
            f"Warning: {missing_mask.sum()} predictions were missing and filled with sample submission values."
        )
        final_sub.loc[missing_mask, "LatitudeDegrees"] = sample_sub.loc[
            missing_mask, "LatitudeDegrees"
        ]
        final_sub.loc[missing_mask, "LongitudeDegrees"] = sample_sub.loc[
            missing_mask, "LongitudeDegrees"
        ]

    # 6. Save Submission
    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"Submission successfully saved to {output_path}")
