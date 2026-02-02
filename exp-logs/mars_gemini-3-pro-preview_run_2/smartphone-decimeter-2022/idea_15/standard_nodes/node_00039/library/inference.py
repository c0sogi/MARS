import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import setup_logger, meters_to_wls
from library.data_loader import DataProcessor, GNSSDataset
from library.model import SARTransformer


def predict(model, test_loader, device):
    """
    Runs inference on the test loader using the given model.

    Args:
        model (nn.Module): The loaded PyTorch model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Array of predictions (N, 2) representing (dLat_m, dLon_m).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle dataset return format (x_kin, x_sky) or (x_kin, x_sky, y)
            if len(batch) == 2:
                x_kin, x_sky = batch
            else:
                x_kin, x_sky, _ = batch

            x_kin = x_kin.to(device)
            x_sky = x_sky.to(device)

            preds = model(x_kin, x_sky)
            all_preds.append(preds.cpu().numpy())

    if not all_preds:
        return np.array([])

    return np.vstack(all_preds)


def get_test_indices_and_baseline(processor):
    """
    Reconstructs the mapping of (tripId, timestamp) -> index in the prediction array.
    Also returns the baseline WLS coordinates for all requested test points.

    This mimics the logic in DataProcessor.process_data for test mode to ensure alignment
    between the model predictions and the submission file structure.

    Args:
        processor (DataProcessor): Instance of DataProcessor to use aggregation/windowing logic.

    Returns:
        valid_indices_map (list): List of (tripId, timestamp) tuples matching the order of predictions.
        df_baseline (pd.DataFrame): DataFrame containing baseline WLS coordinates for all requested points.
    """
    # Load test metadata which contains the requested tripIds and timestamps
    df_meta = pd.read_csv(processor.metadata_path)

    valid_indices_map = []  # List of (tripId, timestamp) matching the order of X_kin
    baseline_data = []  # List of dicts with baseline info for all requested points

    # DataProcessor iterates over unique tripIds
    unique_trips = df_meta["tripId"].unique()

    for trip_id in unique_trips:
        trip_meta = df_meta[df_meta["tripId"] == trip_id]
        if trip_meta.empty:
            continue

        # Load GNSS data using path from metadata
        gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            continue

        df_gnss = pd.read_csv(gnss_path)

        # Aggregate raw GNSS to epochs (same as DataProcessor)
        df_agg = processor._aggregate_gnss(df_gnss)

        # --- 1. Build Baseline for ALL requested points in this trip ---
        # Ensure types match for merging
        df_agg["utcTimeMillis"] = df_agg["utcTimeMillis"].astype(np.int64)
        trip_meta_req = trip_meta[["tripId", "UnixTimeMillis"]].copy()
        trip_meta_req["UnixTimeMillis"] = trip_meta_req["UnixTimeMillis"].astype(
            np.int64
        )

        # Merge to get WLS for requested points
        # DataProcessor._aggregate_gnss computes wls_lat/lon/alt
        df_baseline_trip = pd.merge(
            trip_meta_req,
            df_agg[["utcTimeMillis", "wls_lat", "wls_lon"]],
            left_on="UnixTimeMillis",
            right_on="utcTimeMillis",
            how="left",
        )

        # Store baseline rows
        for _, row in df_baseline_trip.iterrows():
            baseline_data.append(
                {
                    "tripId": row["tripId"],
                    "UnixTimeMillis": row["UnixTimeMillis"],
                    "wls_lat": row["wls_lat"],
                    "wls_lon": row["wls_lon"],
                }
            )

        # --- 2. Reconstruct Valid Indices for Model Predictions ---
        # Logic must match DataProcessor._create_windows exactly
        # It creates windows for ALL aggregated epochs, then filters
        _, _, _, indices = processor._create_windows(df_agg, Config.WINDOW_SIZE)

        req_times = set(trip_meta["UnixTimeMillis"].values)

        # Filter indices (timestamps) just like DataProcessor does for 'test' mode
        for t in indices:
            if t in req_times:
                valid_indices_map.append((trip_id, t))

    return valid_indices_map, pd.DataFrame(baseline_data)


def generate_submission(load_cached_data=True):
    """
    Main inference pipeline.

    1. Loads and processes test data.
    2. Loads the trained model.
    3. Runs inference.
    4. Reconstructs absolute coordinates combining baseline WLS and predicted residuals.
    5. Saves the submission file.

    Args:
        load_cached_data (bool): Whether to try loading pre-processed data from cache.
    """
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "inference.log"))
    logger.info("Starting inference pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and Process Data
    logger.info("Processing test data...")
    processor = DataProcessor(mode="test")

    # Check if cache exists, if not force compute
    # This returns the tensors used for model input
    X_kin, X_sky, _ = processor.process_data(load_cached_data=load_cached_data)

    if len(X_kin) == 0:
        logger.warning("No test data processed! Check input directory structure.")
        return

    test_dataset = GNSSDataset(X_kin, X_sky)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Load Model
    logger.info("Loading model...")
    model = SARTransformer(
        kinematic_input_dim=len(Config.KINEMATIC_FEATURES),
        sky_input_dim=len(Config.SKY_FEATURES),
        output_dim=len(Config.TARGET_COLS),
    )

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 3. Run Inference
    logger.info("Running prediction...")
    preds_arr = predict(model, test_loader, device)
    logger.info(f"Predictions shape: {preds_arr.shape}")

    # 4. Reconstruct Coordinates
    logger.info("Reconstructing coordinates and aligning with submission format...")

    # We need to map the flat array of predictions back to (tripId, timestamp)
    # We re-run the index generation logic to ensure perfect alignment
    valid_indices_map, df_baseline = get_test_indices_and_baseline(processor)

    if len(preds_arr) != len(valid_indices_map):
        raise ValueError(
            f"Mismatch between predictions ({len(preds_arr)}) and reconstructed indices ({len(valid_indices_map)}). Data processing logic might have diverged."
        )

    # Create a map for fast lookup of predictions: (tripId, timestamp) -> (dLat_m, dLon_m)
    pred_lookup = {}
    for i, (trip_id, ts) in enumerate(valid_indices_map):
        pred_lookup[(trip_id, ts)] = preds_arr[i]

    # 5. Build Submission DataFrame
    # We iterate over the baseline dataframe which contains ALL requested test points.
    # If a prediction exists (window was valid), we use it.
    # If not (e.g. start/end of trip dropped by windowing), we fall back to baseline WLS (residual=0).

    final_lats = []
    final_lons = []

    # Counters for stats
    predicted_count = 0
    fallback_count = 0

    for _, row in df_baseline.iterrows():
        trip_id = row["tripId"]
        ts = int(row["UnixTimeMillis"])
        wls_lat = row["wls_lat"]
        wls_lon = row["wls_lon"]

        # Check if we have a valid WLS baseline
        if pd.isna(wls_lat) or pd.isna(wls_lon):
            # Fallback to 0.0 if baseline is missing (should not happen in valid data)
            final_lats.append(0.0)
            final_lons.append(0.0)
            fallback_count += 1
            continue

        if (trip_id, ts) in pred_lookup:
            d_lat_m, d_lon_m = pred_lookup[(trip_id, ts)]
            predicted_count += 1
        else:
            # Fallback to baseline (0 correction)
            d_lat_m, d_lon_m = 0.0, 0.0
            fallback_count += 1

        # Convert predicted metric residuals to degrees and add to baseline
        lat, lon = meters_to_wls(wls_lat, wls_lon, d_lat_m, d_lon_m)
        final_lats.append(lat)
        final_lons.append(lon)

    df_submission = df_baseline[["tripId", "UnixTimeMillis"]].copy()
    df_submission["LatitudeDegrees"] = final_lats
    df_submission["LongitudeDegrees"] = final_lons

    logger.info(
        f"Submission stats: Predicted={predicted_count}, Fallback (Baseline)={fallback_count}"
    )

    # Save
    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
