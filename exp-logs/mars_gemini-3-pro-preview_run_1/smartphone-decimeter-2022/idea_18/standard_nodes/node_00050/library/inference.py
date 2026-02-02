import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.preprocessing import PreProcessor
from library.dataset import GnssSequenceDataset
from library.model import SEResUNet1D


def predict_drive(debug=False):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    print("Loading and processing test data...")
    preprocessor = PreProcessor()
    # We only need the test dataframe
    _, _, test_df = preprocessor.process_data(load_cached_data=True)

    if debug:
        print("Debug mode: Sampling test data...")
        # Sample a few drives
        drives = test_df["drive_id"].unique()[:2]
        test_df = test_df[test_df["drive_id"].isin(drives)].copy()

    # Create Dataset and DataLoader
    # mode='test' ensures full sequences are returned without windowing/padding logic intended for training
    test_dataset = GnssSequenceDataset(
        test_df,
        mode="test",
        window_size=Config.TRAIN_WINDOW_SIZE,  # Not used for splitting in test mode
    )

    # Batch size must be 1 because sequences have variable lengths
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test batches: {len(test_loader)}")

    # 3. Model Loading
    model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading model from {model_path}...")
    model = SEResUNet1D(in_channels=Config.INPUT_CHANNELS, out_channels=2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (features, _, meta) in enumerate(test_loader):
            features = features.to(device)

            # Forward pass
            # In eval mode, the model returns only the final head output
            # Output shape: (B, 2, T) -> (1, 2, T)
            preds = model(features)

            # Move to CPU and numpy
            # Permute to (T, 2) for easier processing
            pred_enu = preds.cpu().numpy()[0].T
            pred_e = pred_enu[:, 0]
            pred_n = pred_enu[:, 1]

            # Extract Metadata
            # meta items are batched (size 1), so we take index 0
            # Timestamps are Int64, need to be careful with tensor conversion if it happened
            timestamps = meta["timestamp"][0].numpy()
            drive_id = meta["drive_id"][0]
            phone_name = meta["phone_name"][0]

            # Baseline WLS positions
            base_lat = meta["baseline_lat"][0].numpy()
            base_lon = meta["baseline_lon"][0].numpy()

            # 5. Coordinate Conversion (ENU Offsets -> Lat/Lon Degrees)
            # Using the same approximation as in training
            # 1 deg lat approx 111320m
            # 1 deg lon approx 111320m * cos(lat)
            lat_scale = 111320.0
            lon_scale = 111320.0 * np.cos(np.radians(base_lat))

            pred_lat = base_lat + (pred_n / lat_scale)
            pred_lon = base_lon + (pred_e / lon_scale)

            # 6. Collect Results
            # We need to reconstruct the tripId for submission matching
            trip_id = f"{drive_id}-{phone_name}"

            batch_results = pd.DataFrame(
                {
                    "tripId": trip_id,
                    "UnixTimeMillis": timestamps,
                    "LatitudeDegrees": pred_lat,
                    "LongitudeDegrees": pred_lon,
                }
            )

            results.append(batch_results)

    # 7. Generate Submission File
    print("Generating submission file...")
    if not results:
        print("No predictions generated.")
        return

    all_predictions = pd.concat(results, ignore_index=True)

    # Load sample submission to ensure correct format and rows
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # FIX: Round sample_sub timestamps for merging (Cite debug_lesson_14)
    sample_sub["UnixTimeMillis_Rounded"] = (
        np.round(sample_sub["UnixTimeMillis"] / 1000.0) * 1000.0
    )
    sample_sub["UnixTimeMillis_Rounded"] = sample_sub["UnixTimeMillis_Rounded"].astype(
        np.int64
    )

    # Merge predictions onto sample submission using rounded timestamp
    submission = pd.merge(
        sample_sub,
        all_predictions,
        left_on=["tripId", "UnixTimeMillis_Rounded"],
        right_on=["tripId", "UnixTimeMillis"],
        how="left",
        suffixes=("", "_pred"),
    )

    # Use predicted values where available
    submission["LatitudeDegrees"] = submission["LatitudeDegrees_pred"]
    submission["LongitudeDegrees"] = submission["LongitudeDegrees_pred"]

    # Fill missing values (if any) with WLS baseline from test_df
    if submission["LatitudeDegrees"].isnull().any():
        print("Warning: Some predictions are missing. Filling with WLS baseline.")

        if "wls_lat" in test_df.columns:
            # Create lookup from test_df (which has rounded timestamps)
            wls_lookup = test_df.copy()
            wls_lookup["tripId"] = (
                wls_lookup["drive_id"] + "-" + wls_lookup["phone_name"]
            )

            # Merge WLS info using rounded timestamps
            submission = pd.merge(
                submission,
                wls_lookup[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]],
                left_on=["tripId", "UnixTimeMillis_Rounded"],
                right_on=["tripId", "UnixTimeMillis"],
                how="left",
                suffixes=("", "_wls"),
            )

            submission["LatitudeDegrees"] = submission["LatitudeDegrees"].fillna(
                submission["wls_lat"]
            )
            submission["LongitudeDegrees"] = submission["LongitudeDegrees"].fillna(
                submission["wls_lon"]
            )

    # Final cleanup
    submission = submission[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    # Final check
    assert len(submission) == len(
        sample_sub
    ), f"Submission length mismatch! Expected {len(sample_sub)}, got {len(submission)}"

    # Save
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())
