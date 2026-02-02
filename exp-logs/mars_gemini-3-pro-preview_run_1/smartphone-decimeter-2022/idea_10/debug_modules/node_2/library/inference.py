import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import enu_to_ecef, ecef_to_llh
from library.data_preprocessing import get_data
from library.dataset import GNSSSequenceDataset, collate_padded_sequences
from library.model import TransUNet1D


def get_wls_baseline(drive_id, phone_name, timestamps):
    """
    Loads WLS baseline coordinates from device_gnss.csv for a specific drive/phone
    and aligns them with the requested timestamps.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        timestamps (np.array): Array of rounded timestamps (UnixTimeMillis) required.

    Returns:
        pd.DataFrame: DataFrame containing aligned WLS coordinates [UnixTimeMillis, WlsPositionXEcefMeters, ...].
    """
    gnss_path = os.path.join(
        Config.INPUT_DIR, "test", drive_id, phone_name, "device_gnss.csv"
    )

    if not os.path.exists(gnss_path):
        # Fallback if file not found (should not happen in valid test set)
        return pd.DataFrame()

    # Load WLS positions
    cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df_gnss = pd.read_csv(gnss_path, usecols=cols)

    # Quantize timestamps to match the model's input resolution
    df_gnss["UnixTimeMillis"] = (df_gnss["utcTimeMillis"] + 500) // 1000 * 1000

    # Drop duplicates to ensure unique baseline per second (taking the first valid fix)
    df_gnss = df_gnss.drop_duplicates(subset=["UnixTimeMillis"]).dropna()

    # Create a dataframe for the requested timestamps
    df_req = pd.DataFrame({"UnixTimeMillis": timestamps})

    # Merge to align. Use left join to keep all requested timestamps.
    df_merged = pd.merge(df_req, df_gnss, on="UnixTimeMillis", how="left")

    # Interpolate missing WLS positions (if any gaps exist in raw log but prediction is required)
    # ECEF coordinates vary smoothly, so linear interpolation is acceptable for small gaps
    pos_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df_merged[pos_cols] = df_merged[pos_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # Fill remaining NaNs (e.g., at edges) with 0 or forward/backward fill if interpolate failed
    df_merged[pos_cols] = (
        df_merged[pos_cols].fillna(method="bfill").fillna(method="ffill").fillna(0)
    )

    return df_merged


def generate_predictions(model_path=None, load_cached_data=True, batch_size=1):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights. If None, uses default checkpoint.
        load_cached_data (bool): Whether to use cached preprocessed features.
        batch_size (int): Batch size for inference.
    """
    print("Initializing Inference Pipeline...")

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Preserve original timestamps for submission
    # The model works on rounded timestamps (1Hz), so we map rounded -> original(s)
    # Note: Multiple original timestamps might map to one rounded timestamp if sampling > 1Hz,
    # but the competition is typically 1Hz. We keep track of the mapping.
    df_test_meta["OriginalTimeMillis"] = df_test_meta["UnixTimeMillis"]
    df_test_meta["UnixTimeMillis"] = (
        (df_test_meta["UnixTimeMillis"] + 500) // 1000 * 1000
    )

    # 2. Load Features
    # get_data will merge features onto the metadata based on rounded timestamps
    df_test_data = get_data(df_test_meta, load_cached_data=load_cached_data)

    # 3. Prepare Dataset and Loader
    # Scaler is loaded from the training directory
    dataset = GNSSSequenceDataset(
        df_test_data,
        feature_cols=Config.INPUT_FEATURES,
        target_cols=None,
        mode="test",
        scaler_dir=Config.WORKING_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded_sequences,
        num_workers=2,
    )

    # 4. Load Model
    device = Config.DEVICE
    model = TransUNet1D().to(device)

    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # 5. Inference Loop
    results = []

    with torch.no_grad():
        for batch_idx, (features, _, mask, meta_list) in enumerate(loader):
            features = features.to(device)
            mask = mask.to(device)

            # Forward pass: (B, Output_Dim, L)
            outputs = model(features, mask)

            # Permute to (B, L, Output_Dim)
            outputs = outputs.permute(0, 2, 1).cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)

            # Process each sequence in the batch
            for i in range(len(meta_list)):
                drive_id = meta_list[i]["drive_id"]
                phone_name = meta_list[i]["phone_name"]

                # Get valid sequence length
                valid_len = np.sum(mask_np[i])
                if valid_len == 0:
                    continue

                # Extract predictions (dLat_meters, dLon_meters)
                # Shape: (L, 2)
                pred_residuals = outputs[i, :valid_len, :]

                # Get timestamps from metadata (these are the rounded ones used for sorting)
                seq_timestamps = meta_list[i]["timestamps"]

                # Retrieve Baseline WLS
                df_wls = get_wls_baseline(drive_id, phone_name, seq_timestamps)

                if df_wls.empty or len(df_wls) != len(seq_timestamps):
                    # Should not happen with interpolation, but safety check
                    print(
                        f"Warning: WLS mismatch for {drive_id}-{phone_name}. Filling with 0."
                    )
                    wls_x = np.zeros(len(seq_timestamps))
                    wls_y = np.zeros(len(seq_timestamps))
                    wls_z = np.zeros(len(seq_timestamps))
                else:
                    wls_x = df_wls["WlsPositionXEcefMeters"].values
                    wls_y = df_wls["WlsPositionYEcefMeters"].values
                    wls_z = df_wls["WlsPositionZEcefMeters"].values

                # Convert Baseline ECEF -> LLH
                wls_lat, wls_lon, wls_alt = ecef_to_llh(wls_x, wls_y, wls_z)

                # Apply Predicted Residuals (ENU -> ECEF)
                # pred_residuals[:, 0] is dNorth (dLat_meters)
                # pred_residuals[:, 1] is dEast (dLon_meters)
                d_north = pred_residuals[:, 0]
                d_east = pred_residuals[:, 1]
                d_up = np.zeros_like(d_north)  # Assume 0 vertical correction

                pred_x, pred_y, pred_z = enu_to_ecef(
                    d_east, d_north, d_up, wls_lat, wls_lon, wls_alt
                )

                # Convert Corrected ECEF -> LLH
                pred_lat, pred_lon, _ = ecef_to_llh(pred_x, pred_y, pred_z)

                # Map back to Original Timestamps
                # We need to link the predictions (on rounded times) to the original requested times.
                # Filter the main test dataframe for this drive/phone
                df_drive_orig = df_test_meta[
                    (df_test_meta["drive_id"] == drive_id)
                    & (df_test_meta["phone_name"] == phone_name)
                ].copy()

                # Create a lookup for predictions
                # df_preds has columns: UnixTimeMillis (rounded), Lat, Lon
                df_preds = pd.DataFrame(
                    {
                        "UnixTimeMillis": seq_timestamps,
                        "PredLat": pred_lat,
                        "PredLon": pred_lon,
                    }
                )

                # Merge original requests with predictions on rounded time
                # Note: df_drive_orig has 'UnixTimeMillis' already rounded from step 1,
                # and 'OriginalTimeMillis' preserved.
                df_final = pd.merge(
                    df_drive_orig, df_preds, on="UnixTimeMillis", how="left"
                )

                # Fill missing predictions (if any) with WLS or 0 (fallback)
                # In a robust pipeline, we might interpolate, but here we expect full coverage
                df_final["PredLat"] = df_final["PredLat"].fillna(0)
                df_final["PredLon"] = df_final["PredLon"].fillna(0)

                # Collect results
                results.append(
                    df_final[["tripId", "OriginalTimeMillis", "PredLat", "PredLon"]]
                )

    # 6. Construct Submission
    submission_df = pd.concat(results, ignore_index=True)

    # Rename columns to match submission format
    submission_df = submission_df.rename(
        columns={
            "OriginalTimeMillis": "UnixTimeMillis",
            "PredLat": "LatitudeDegrees",
            "PredLon": "LongitudeDegrees",
        }
    )

    # Sort by tripId and Time
    submission_df = submission_df.sort_values(["tripId", "UnixTimeMillis"])

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Total predictions: {len(submission_df)}")
