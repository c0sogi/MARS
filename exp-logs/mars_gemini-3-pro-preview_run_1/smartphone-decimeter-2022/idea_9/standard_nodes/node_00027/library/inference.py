import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import HybridResUNetGRU
from library.dataset import get_dataloaders
from library.utils import cartesian_to_wgs84


def predict_drive(model, features, device):
    """
    Runs the trained model on a batch of drive features to predict Cartesian offsets.

    Args:
        model (nn.Module): The trained HybridResUNetGRU model.
        features (torch.Tensor): Input features tensor of shape (Batch, Channels, Time).
        device (torch.device): The device (CPU/GPU) to run inference on.

    Returns:
        torch.Tensor: Predicted offsets (North, East) of shape (Batch, 2, Time).
    """
    model.eval()
    features = features.to(device)
    with torch.no_grad():
        outputs = model(features)
    return outputs


def generate_submission(load_cached_data=True, debug=False):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
                                 If False, recomputes features from raw logs.
        debug (bool): If True, runs on a small subset of the test data for debugging purposes.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # Handle debug mode by temporarily modifying Config
    # This allows get_dataloaders to respect the debug flag without modifying library code
    original_debug_state = Config.DEBUG
    if debug:
        Config.DEBUG = True
        print("Running in DEBUG mode (subset of data).")

    try:
        # Load Test Data
        # get_dataloaders returns (train, val, test). We only need the test loader.
        print("Loading test data...")
        _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)
    except Exception as e:
        print(f"Error loading data: {e}")
        Config.DEBUG = original_debug_state
        return
    finally:
        # Restore original configuration
        Config.DEBUG = original_debug_state

    # Initialize Model
    model = HybridResUNetGRU().to(device)

    # Load Model Weights
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(
            f"Error: Model weights not found at {model_path}. Please train the model first."
        )
        return

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = []
    print(f"Starting inference on {len(test_loader.dataset)} drives...")

    # Inference Loop
    for batch_idx, batch in enumerate(test_loader):
        features = batch["features"]
        baselines = batch["baseline"]
        timestamps = batch["timestamps"]
        drive_ids = batch["drive_id"]
        phone_names = batch["phone_name"]

        # Predict Cartesian Offsets (North, East)
        outputs = predict_drive(model, features, device)  # Shape: (Batch, 2, Time)

        batch_size = features.size(0)

        # Process each drive in the batch
        for i in range(batch_size):
            # The collate function pads sequences. We must slice to the valid length.
            # Timestamps list contains the valid timestamps for this specific drive.
            valid_length = len(timestamps[i])

            # Slice prediction: (2, T_padded) -> (2, T_valid) -> Transpose to (T_valid, 2)
            # Index 0 is North, Index 1 is East
            pred_offsets = outputs[i, :, :valid_length].cpu().numpy().T

            # Get WLS Baseline: (T_valid, 2) where col 0 is Lat, col 1 is Lon
            base_pos = baselines[i]

            # Reconstruct WGS84 Coordinates
            # lat = base_lat + (north / LAT_TO_M)
            # lon = base_lon + (east / LON_TO_M)
            pred_lat, pred_lon = cartesian_to_wgs84(
                north=pred_offsets[:, 0],
                east=pred_offsets[:, 1],
                lat_ref=base_pos[:, 0],
                lon_ref=base_pos[:, 1],
            )

            # Format trip identifier
            trip_id = f"{drive_ids[i]}-{phone_names[i]}"

            # Aggregate results
            for t, lat, lon in zip(timestamps[i], pred_lat, pred_lon):
                results.append(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": t,
                        "LatitudeDegrees": lat,
                        "LongitudeDegrees": lon,
                    }
                )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns match the required submission format
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not submission_df.empty:
        submission_df = submission_df[required_cols]
    else:
        submission_df = pd.DataFrame(columns=required_cols)

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)

    print(f"Submission generated successfully.")
    print(f"Saved to: {out_path}")
    print(f"Total predictions: {len(submission_df)}")
