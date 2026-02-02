import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import Meters_to_WGS84
from library.data_preprocessing import prepare_test_data
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import AtrousResUNet


def predict_and_convert(
    device, scaler, model_path=None, load_cached_data=True, debug_size=None
):
    """
    Loads the trained model, runs inference on the test dataset, converts predictions
    to WGS84 coordinates, and generates the submission file.

    Args:
        device (torch.device): Device to run inference on.
        scaler (dict): Dictionary containing 'mean' and 'std' for feature normalization.
        model_path (str, optional): Path to the saved model state dict. Defaults to Config.WORKING_DIR/best_model.pth.
        load_cached_data (bool): Whether to load pre-processed test data from cache.
        debug_size (int, optional): Limit the number of test drives for debugging.

    Returns:
        pd.DataFrame: The final submission dataframe.
    """

    # Handle debug size by updating Config temporarily
    if debug_size is not None:
        Config.DEBUG_SAMPLE_SIZE = debug_size
        # Force re-processing if debugging to ensure correct size
        load_cached_data = False
        print(f"Debug mode enabled: processing {debug_size} drives.")

    # 1. Prepare Test Data
    print("Preparing test data...")
    test_df = prepare_test_data(load_cached_data=load_cached_data)

    # Create Dataset and Loader
    # We use the scaler provided (computed from training data) to normalize test data
    test_dataset = GNSSSequenceDataset(test_df, mode="test", scaler=scaler)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one trip at a time for robust reconstruction
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Load Model
    print("Loading model...")
    model = AtrousResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_dim=Config.HIDDEN_DIM,
    ).to(device)

    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference Loop
    results = []
    print("Running inference...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            masks = batch["masks"]

            # Forward pass
            outputs = model(features)
            # outputs is a list [final, aux1, aux2], we want final output at index 0
            final_out = outputs[0].cpu().numpy()  # Shape: (B, 2, L)

            # Reconstruct per sequence in batch (batch_size is 1 here)
            for i in range(features.shape[0]):
                trip_id = batch["trip_ids"][i]
                wls_pos = batch["wls_pos"][i]  # (L, 2) -> lat, lon
                timestamps = batch["timestamps"][i]

                # Get valid length from mask
                valid_len = int(masks[i].sum().item())

                if valid_len == 0:
                    continue

                # Extract predictions for valid steps: (2, valid_len) -> (valid_len, 2)
                preds = final_out[i, :, :valid_len].T

                delta_north = preds[:, 0]
                delta_east = preds[:, 1]

                base_lat = wls_pos[:valid_len, 0]
                base_lon = wls_pos[:valid_len, 1]
                valid_timestamps = timestamps[:valid_len]

                # Convert offsets to WGS84
                pred_lat, pred_lon = Meters_to_WGS84(
                    base_lat, base_lon, delta_north, delta_east
                )

                # Create DataFrame for this trip
                trip_res = pd.DataFrame(
                    {
                        "tripId": [trip_id] * valid_len,
                        "UnixTimeMillis": valid_timestamps,
                        "LatitudeDegrees": pred_lat,
                        "LongitudeDegrees": pred_lon,
                    }
                )

                results.append(trip_res)

    # 4. Generate Submission File
    if not results:
        print("No predictions generated.")
        return pd.DataFrame()

    submission_df = pd.concat(results, ignore_index=True)

    # Load sample submission template to ensure correct format and rows
    if not os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {Config.SAMPLE_SUBMISSION_PATH}"
        )

    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Create merge keys
    submission_df["key"] = (
        submission_df["tripId"] + "_" + submission_df["UnixTimeMillis"].astype(str)
    )
    sample_sub["key"] = (
        sample_sub["tripId"] + "_" + sample_sub["UnixTimeMillis"].astype(str)
    )

    # Merge predictions into sample submission structure
    final_sub = sample_sub.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]).merge(
        submission_df[["key", "LatitudeDegrees", "LongitudeDegrees"]],
        on="key",
        how="left",
    )

    # Handle missing predictions (if any) via interpolation
    # This handles cases where test_metadata might have had gaps or model output was masked
    if final_sub["LatitudeDegrees"].isnull().any():
        print("Interpolating missing predictions...")
        final_sub["LatitudeDegrees"] = (
            final_sub["LatitudeDegrees"]
            .interpolate(method="linear")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
        final_sub["LongitudeDegrees"] = (
            final_sub["LongitudeDegrees"]
            .interpolate(method="linear")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

    final_sub = final_sub.drop(columns=["key"])

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return final_sub
