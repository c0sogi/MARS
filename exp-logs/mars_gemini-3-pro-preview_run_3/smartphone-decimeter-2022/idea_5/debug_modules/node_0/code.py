import pandas as pd
import numpy as np
import os
import sys
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library modules can be imported
sys.path.append(".")

# Import library modules
from library.config import Config
from library.utils import get_logger, calc_score
from library.data_loader import load_metadata, load_drive_data
from library.feature_eng import create_features, compute_targets
from library.model import LGBMRegressorWrapper
from library.postprocess import convert_enu_to_geodetic, apply_kalman_smoothing

# Setup Logger
logger = get_logger("Demo_Script")


def main():
    logger.info("Starting GNSS Positioning Demo Script")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We modify the global Config to use fewer estimators and a specific working dir
    # to ensure the demo runs quickly and doesn't overwrite production models.
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 16
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure demo working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading (Subset)
    # -------------------------------------------------------------------------
    logger.info("Loading metadata...")
    try:
        train_meta = load_metadata("train")
        val_meta = load_metadata("val")
    except FileNotFoundError as e:
        logger.error(f"Metadata not found: {e}")
        return

    # Select a small subset of drives to process to keep execution time low
    # We pick 1 drive for training and 1 for validation
    train_drives = train_meta["drive_id"].unique()[:1]
    val_drives = val_meta["drive_id"].unique()[:1]

    logger.info(f"Selected Train Drives: {train_drives}")
    logger.info(f"Selected Val Drives: {val_drives}")

    def load_and_process_subset(metadata, drives, desc):
        """
        Helper function to load raw data, create features, and compute targets
        for a list of drives.
        """
        data_list = []
        # Get unique phone/drive combos
        trips = metadata[metadata["drive_id"].isin(drives)][
            ["drive_id", "phone_name"]
        ].drop_duplicates()

        if trips.empty:
            logger.warning(f"No trips found for {desc}")
            return pd.DataFrame()

        for _, row in trips.iterrows():
            d_id = row["drive_id"]
            p_name = row["phone_name"]
            logger.info(f"Processing {desc} trip: {d_id} - {p_name}")

            # Load raw data
            try:
                g_df, i_df = load_drive_data(d_id, p_name, metadata)
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                continue

            # Limit rows for speed if dataset is huge (e.g. > 5k rows)
            # This is just for demonstration purposes
            if len(g_df) > 5000:
                g_df = g_df.iloc[:5000]
                # IMU data is usually higher frequency, cut roughly proportional
                i_df = i_df.iloc[:50000]

            # Feature Engineering
            feats = create_features(g_df, i_df)

            # Compute Targets
            # Check if GT columns exist (they should for train/val)
            if "LatitudeDegrees" in feats.columns:
                feats = compute_targets(feats)

            data_list.append(feats)

        if not data_list:
            return pd.DataFrame()

        return pd.concat(data_list, ignore_index=True)

    logger.info("Preparing Training Data...")
    train_df = load_and_process_subset(train_meta, train_drives, "Train")

    logger.info("Preparing Validation Data...")
    val_df = load_and_process_subset(val_meta, val_drives, "Validation")

    # Assertions to verify data integrity
    assert not train_df.empty, "Training data is empty."
    assert not val_df.empty, "Validation data is empty."
    assert (
        Config.TARGET_EAST in train_df.columns
    ), "Target East column missing in train."
    assert (
        Config.TARGET_NORTH in train_df.columns
    ), "Target North column missing in train."
    assert "Cn0DbHz_mean" in train_df.columns, "Feature Cn0DbHz_mean missing."

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    logger.info("Initializing LightGBM Wrapper...")
    model_wrapper = LGBMRegressorWrapper()

    # Verify params update
    assert model_wrapper.params["n_estimators"] == 10, "Parameter override failed."

    logger.info("Training models...")
    model_wrapper.train(train_df, val_df)

    # Verify model files were created
    assert os.path.exists(model_wrapper.model_path_east), "East model file missing."
    assert os.path.exists(model_wrapper.model_path_north), "North model file missing."

    # -------------------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------------------
    logger.info("Running inference on Validation set...")
    preds = model_wrapper.predict(val_df)

    assert len(preds) == len(val_df), "Prediction count mismatch."
    assert Config.TARGET_EAST in preds.columns, "Prediction column missing."

    # -------------------------------------------------------------------------
    # 5. Post-processing (ENU -> Geodetic + Smoothing)
    # -------------------------------------------------------------------------
    logger.info("Post-processing predictions...")

    # Prepare input for post-processing
    # We need tripId, UnixTimeMillis, predicted residuals, and reference WLS coords
    # RefLat, RefLon, RefAlt were computed during compute_targets
    post_input = val_df[
        ["tripId", "UnixTimeMillis", "RefLat", "RefLon", "RefAlt"]
    ].copy()
    post_input[Config.TARGET_EAST] = preds[Config.TARGET_EAST]
    post_input[Config.TARGET_NORTH] = preds[Config.TARGET_NORTH]

    # 1. Convert residuals back to Geodetic coordinates
    df_geo = convert_enu_to_geodetic(post_input)
    assert "LatitudeDegrees" in df_geo.columns, "Geodetic conversion failed."

    # 2. Apply Kalman Smoothing
    # Note: This function expects 'x_ecef', 'y_ecef', 'z_ecef' which are created by convert_enu_to_geodetic
    # We use default noise parameters here
    df_smoothed = apply_kalman_smoothing(df_geo, process_noise=0.5, meas_noise=3.0)

    assert (
        "LatitudeDegrees" in df_smoothed.columns
    ), "Smoothing failed to return Latitude."
    assert len(df_smoothed) > 0, "Smoothed dataframe is empty."

    # -------------------------------------------------------------------------
    # 6. Evaluation
    # -------------------------------------------------------------------------
    logger.info("Evaluating performance...")

    # Prepare Ground Truth from val_df
    gt_df = val_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Prepare Predictions
    # df_smoothed contains the final LatitudeDegrees and LongitudeDegrees
    pred_df = df_smoothed[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Calculate Score
    try:
        score = calc_score(pred_df, gt_df)
        logger.info(
            f"Validation Score (Mean 50/95 Percentile Error): {score:.4f} meters"
        )
    except ValueError as e:
        logger.error(f"Scoring failed: {e}")
        # Fallback: Find intersection if timestamps don't align perfectly due to resampling
        common_ids = pd.merge(
            pred_df, gt_df, on=["tripId", "UnixTimeMillis"], how="inner"
        )
        if common_ids.empty:
            logger.error("No overlapping timestamps between predictions and GT.")
        else:
            # Rename columns to match calc_score expectations
            p_df = common_ids[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees_x", "LongitudeDegrees_x"]
            ].rename(
                columns={
                    "LatitudeDegrees_x": "LatitudeDegrees",
                    "LongitudeDegrees_x": "LongitudeDegrees",
                }
            )
            g_df = common_ids[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees_y", "LongitudeDegrees_y"]
            ].rename(
                columns={
                    "LatitudeDegrees_y": "LatitudeDegrees",
                    "LongitudeDegrees_y": "LongitudeDegrees",
                }
            )
            score = calc_score(p_df, g_df)
            logger.info(f"Validation Score (Intersection): {score:.4f} meters")

    # -------------------------------------------------------------------------
    # 7. Generate Submission File (Mock)
    # -------------------------------------------------------------------------
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    pred_df.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved to {submission_path}")

    logger.info("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
