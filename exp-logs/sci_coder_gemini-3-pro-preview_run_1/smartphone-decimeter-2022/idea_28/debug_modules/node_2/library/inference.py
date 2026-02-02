import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library import config
from library.architecture import SEResUNet1D
from library.dataset import GnssSequenceDataset, collate_fn
from library.utils import enu_to_geodetic


def generate_submission(
    load_cached_data=True, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
):
    """
    Generates the submission file for the test set using the trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed test data from cache.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
    """
    device = config.DEVICE
    print(f"Starting inference on device: {device}")

    # 1. Load Model
    print(f"Loading model from {config.MODEL_PATH}...")
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Please train the model first."
        )

    model = SEResUNet1D()
    model.to(device)

    # Load weights
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load Test Data
    print("Loading test dataset...")
    test_dataset = GnssSequenceDataset(split="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    results = []

    # 3. Run Inference
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features = batch["features"].to(device)
            mask = batch[
                "mask"
            ]  # Keep mask on CPU for length calculations if needed, or move if used in model

            # Forward pass
            # features shape: (B, C, L)
            outputs = model(features)

            # Get final head predictions: (B, 2, L)
            # 2 channels: Delta East, Delta North
            preds_enu = outputs["final"].cpu().numpy()

            # Process each sequence in the batch
            batch_size_curr = features.shape[0]
            for i in range(batch_size_curr):
                # Metadata
                drive_id = batch["drive_ids"][i]
                phone_name = batch["phone_names"][i]
                trip_id = f"{drive_id}-{phone_name}"

                # Get valid length from mask
                # mask is (B, L), boolean
                valid_len = mask[i].sum().item()

                # Extract valid predictions
                # preds_enu[i] is (2, L_padded)
                # Slice to valid length and transpose to (L_valid, 2)
                pred_valid = preds_enu[i, :, :valid_len].T
                d_east = pred_valid[:, 0]
                d_north = pred_valid[:, 1]

                # Get Baseline WLS and Timestamps
                # These are numpy arrays of shape (L_valid, ...)
                # Note: dataset returns them unpadded, collate_fn puts them in a list
                wls_lat = batch["wls"][i][:valid_len, 0]
                wls_lon = batch["wls"][i][:valid_len, 1]
                timestamps = batch["timestamps"][i][:valid_len]

                # 4. Reconstruct Coordinates
                pred_lat, pred_lon = enu_to_geodetic(d_east, d_north, wls_lat, wls_lon)

                # Store results
                # We need to append rows for the dataframe
                for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": t,
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # 5. Create Submission File
    print("Formatting submission...")
    df_submission = pd.DataFrame(results)

    # Ensure columns are in the correct order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    df_submission = df_submission[cols]

    # Sort by tripId and Time (optional but good practice)
    df_submission = df_submission.sort_values(["tripId", "UnixTimeMillis"])

    # Save
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    df_submission.to_csv(config.SUBMISSION_PATH, index=False)
    print("Inference complete.")
