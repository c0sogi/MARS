import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import PhaseAwareAttentionResUNet
from library.preprocessing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset
from library.utils import WGS84


def reconstruct_coordinates(wls_lat, wls_lon, delta_north, delta_east):
    """
    Convert predicted metric offsets (North/East) back to Geodetic coordinates (Lat/Lon)
    relative to the WLS baseline.

    Args:
        wls_lat (np.array): Baseline WLS Latitude in degrees.
        wls_lon (np.array): Baseline WLS Longitude in degrees.
        delta_north (np.array): Predicted North offset in meters.
        delta_east (np.array): Predicted East offset in meters.

    Returns:
        pred_lat (np.array): Reconstructed Latitude in degrees.
        pred_lon (np.array): Reconstructed Longitude in degrees.
    """
    # WGS84 Constants
    a = WGS84.a
    e2 = WGS84.e2

    # Convert latitude to radians for curvature calculation
    lat_rad = np.radians(wls_lat)
    sin_lat = np.sin(lat_rad)

    # Radius of curvature in the prime vertical
    Rn = a / np.sqrt(1 - e2 * sin_lat**2)

    # Radius of curvature in the meridian
    Rm = (a * (1 - e2)) / (1 - e2 * sin_lat**2) ** 1.5

    # Convert meters to degrees
    dlat = np.degrees(delta_north / Rm)
    dlon = np.degrees(delta_east / (Rn * np.cos(lat_rad)))

    pred_lat = wls_lat + dlat
    pred_lon = wls_lon + dlon

    return pred_lat, pred_lon


def predict_drive(model, batch, device):
    """
    Runs inference on a single drive batch.

    Args:
        model (nn.Module): Trained model.
        batch (dict): Batch dictionary from GnssSequenceDataset.
        device (torch.device): Device to run inference on.

    Returns:
        preds (np.array): Predicted offsets [Length, 2] (DeltaNorth, DeltaEast).
    """
    features = batch["features"].to(device)
    original_length = batch["original_length"].item()

    # Forward pass
    # Model returns [Batch, Channels, Length]
    # In eval mode, it returns just the final output tensor
    with torch.no_grad():
        outputs = model(features)

    # Unpad sequence to original length
    # Transpose to [Length, Channels] for easier handling
    # Batch size is assumed to be 1 for test inference
    preds = outputs[0, :, :original_length].cpu().numpy().transpose(1, 0)

    return preds


def generate_submission(config=None, load_cached_data=True):
    """
    Generates the submission file for the test set.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to load preprocessed test data from cache.
    """
    if config is None:
        config = Config()

    device = torch.device(config.DEVICE)
    print(f"Generating submission on device: {device}")

    # 1. Load Model
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = PhaseAwareAttentionResUNet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Test Data
    preprocessor = GNSSPreprocessor()
    test_df = preprocessor.generate_dataset(
        split="test", load_cached_data=load_cached_data
    )

    test_dataset = GnssSequenceDataset(test_df, split="test", config=config)

    # DataLoader
    # Batch size 1 because sequences have variable lengths in test mode
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    results = []

    print(f"Running inference on {len(test_dataset)} drives...")

    # 3. Inference Loop
    for batch in test_loader:
        # Run prediction
        preds = predict_drive(model, batch, device)

        # Extract metadata for reconstruction
        # wls_coords shape: (1, Length, 2) -> (Length, 2)
        wls_coords = batch["wls_coords"].numpy()[0]
        timestamps = batch["timestamps"].numpy()[0]
        drive_id = batch["drive_id"][0]
        phone_name = batch["phone_name"][0]

        # Unpack coordinates
        wls_lat = wls_coords[:, 0]
        wls_lon = wls_coords[:, 1]
        delta_north = preds[:, 0]
        delta_east = preds[:, 1]

        # Reconstruct Global Coordinates
        pred_lat, pred_lon = reconstruct_coordinates(
            wls_lat, wls_lon, delta_north, delta_east
        )

        # Construct Trip ID
        # Format matches sample_submission.csv: drive_id-phone_name
        trip_id = f"{drive_id}-{phone_name}"

        # Collect results
        for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
            results.append(
                {
                    "tripId": trip_id,
                    "UnixTimeMillis": t,
                    "LatitudeDegrees": lat,
                    "LongitudeDegrees": lon,
                }
            )

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # 5. Merge with Sample Submission
    # This ensures we have exactly the rows required by the competition
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")

    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)

        # Left merge on sample_sub to keep its structure and order
        final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
            submission_df, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Check for missing predictions
        if final_sub["LatitudeDegrees"].isnull().any():
            print(
                "Warning: Some timestamps in sample_submission were not predicted. Filling with 0.0."
            )
            final_sub = final_sub.fillna(0.0)
    else:
        print(
            "Warning: sample_submission.csv not found. Saving generated predictions directly."
        )
        final_sub = submission_df

    # 6. Save
    final_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
