import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, meters_to_latlon
from library.data_loader import load_data, GNSSWindowDataset, ecef_to_lla
from library.model import SkyStateTransformer

# Initialize logger
logger = get_logger("inference")


def get_wls_baseline(trip_id, timestamps, gnss_rel_path):
    """
    Extracts WLS baseline positions for specific timestamps from a GNSS file.
    Used as a fallback for edge cases where the window-based model cannot predict.
    """
    gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)
    if not os.path.exists(gnss_path):
        logger.warning(f"GNSS file not found: {gnss_path}")
        return pd.DataFrame()

    # Read GNSS data
    df_gnss = pd.read_csv(gnss_path)

    # Filter for the required timestamps
    # Note: GNSS log timestamps might not match exactly, but usually do in this dataset.
    # We aggregate by epoch to get one WLS position per timestamp.
    df_epoch = (
        df_gnss[df_gnss["utcTimeMillis"].isin(timestamps)]
        .groupby("utcTimeMillis")
        .agg(
            {
                "WlsPositionXEcefMeters": "first",
                "WlsPositionYEcefMeters": "first",
                "WlsPositionZEcefMeters": "first",
            }
        )
        .reset_index()
    )

    if df_epoch.empty:
        return pd.DataFrame()

    # Convert ECEF to LLA
    lat, lon, _ = ecef_to_lla(
        df_epoch["WlsPositionXEcefMeters"].values,
        df_epoch["WlsPositionYEcefMeters"].values,
        df_epoch["WlsPositionZEcefMeters"].values,
    )

    return pd.DataFrame(
        {
            "tripId": trip_id,
            "UnixTimeMillis": df_epoch["utcTimeMillis"],
            "LatitudeDegrees": lat,
            "LongitudeDegrees": lon,
        }
    )


def generate_predictions(load_cached_data=True, batch_size=Config.BATCH_SIZE * 2):
    """
    Generates predictions for the test set.

    1. Loads processed test data (windows).
    2. Runs inference using the trained Sky-State Transformer.
    3. Reconstructs absolute coordinates from predicted metric residuals.
    4. Handles edge cases (start/end of trips) by falling back to WLS baseline.
    5. Saves the final submission file.
    """
    logger.info("Starting inference pipeline...")

    # 1. Load Test Data (Processed Windows)
    # This returns only the windows that could be fully formed (excluding edges)
    logger.info(f"Loading test data (Cached: {load_cached_data})...")
    (_, _, test_data) = load_data(load_cached_data=load_cached_data)
    test_X_seq, test_X_sky, test_meta_processed = test_data

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = SkyStateTransformer().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    logger.info(f"Loading model from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Run Inference
    logger.info("Running inference on processed windows...")
    test_dataset = GNSSWindowDataset(test_X_seq, test_X_sky, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    with torch.no_grad():
        for batch_seq, batch_sky in tqdm(test_loader, desc="Predicting"):
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)

            outputs = model(batch_seq, batch_sky)
            all_preds.append(outputs.cpu().numpy())

    predictions_meters = np.concatenate(all_preds, axis=0)

    # 4. Reconstruction (Model Predictions)
    # Convert predicted metric residuals (East, North) back to Lat/Lon
    wls_lat = test_meta_processed["WlsLat"].values
    wls_lon = test_meta_processed["WlsLon"].values

    delta_east = predictions_meters[:, 0]
    delta_north = predictions_meters[:, 1]

    pred_lat, pred_lon = meters_to_latlon(delta_north, delta_east, wls_lat, wls_lon)

    # Create DataFrame for model predictions
    model_preds_df = pd.DataFrame(
        {
            "tripId": test_meta_processed["tripId"],
            "UnixTimeMillis": test_meta_processed["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # 5. Merge with Full Submission Template and Handle Missing Edges
    logger.info("Merging predictions and handling edge cases...")

    # Load the full list of required predictions
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    full_sub_df = pd.read_csv(sample_sub_path, usecols=["tripId", "UnixTimeMillis"])

    # Merge model predictions
    # We use 'left' merge to keep all required rows from sample_submission
    final_df = pd.merge(
        full_sub_df, model_preds_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Check for missing values (rows where model didn't produce a prediction due to windowing)
    missing_mask = final_df["LatitudeDegrees"].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        logger.info(
            f"Found {missing_count} missing predictions (edge cases). Filling with WLS baseline..."
        )

        # Load test metadata to get file paths for missing trips
        test_metadata_full = pd.read_csv(Config.TEST_METADATA_PATH)

        # Identify trips that have missing values
        missing_trips = final_df[missing_mask]["tripId"].unique()

        fallback_rows = []

        for trip_id in tqdm(missing_trips, desc="Filling missing"):
            # Get timestamps needed for this trip
            trip_missing_mask = (final_df["tripId"] == trip_id) & missing_mask
            needed_timestamps = final_df.loc[trip_missing_mask, "UnixTimeMillis"].values

            if len(needed_timestamps) == 0:
                continue

            # Get GNSS path
            trip_info = test_metadata_full[
                test_metadata_full["tripId"] == trip_id
            ].iloc[0]
            gnss_path = trip_info["gnss_path"]

            # Extract WLS
            wls_df = get_wls_baseline(trip_id, needed_timestamps, gnss_path)

            if not wls_df.empty:
                fallback_rows.append(wls_df)

        if fallback_rows:
            fallback_df = pd.concat(fallback_rows, ignore_index=True)

            # Update final dataframe with fallback values
            # We set the index to (tripId, UnixTimeMillis) for easy alignment
            final_df_indexed = final_df.set_index(["tripId", "UnixTimeMillis"])
            fallback_df_indexed = fallback_df.set_index(["tripId", "UnixTimeMillis"])

            final_df_indexed.update(fallback_df_indexed)
            final_df = final_df_indexed.reset_index()

    # 6. Save Submission
    # Ensure no NaNs remain (simple forward/backward fill as last resort if WLS also failed)
    if final_df.isnull().values.any():
        logger.warning("NaNs detected after WLS fallback. Applying ffill/bfill.")
        final_df = (
            final_df.groupby("tripId")
            .apply(lambda x: x.ffill().bfill())
            .reset_index(drop=True)
        )

    final_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission generated and saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Total rows: {len(final_df)}")

    return final_df
