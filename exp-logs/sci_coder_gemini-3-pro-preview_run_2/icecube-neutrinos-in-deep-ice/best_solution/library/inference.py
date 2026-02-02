import os
import gc
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    TEST_META_PATH,
    SUBMISSION_PATH,
    FEATURE_NAMES,
    SEED,
)
from library.data_loader import IceCubeFeatureGenerator
from library.model import GradientBoostingVectorRegressor
from library.utils import setup_logger


def generate_submission(
    test_meta_path: str = TEST_META_PATH,
    output_path: str = SUBMISSION_PATH,
    debug_sample_batches: int = None,
):
    """
    Generates the submission file by iterating through test batches,
    creating features, predicting, and writing to CSV incrementally.

    Args:
        test_meta_path (str): Path to the test metadata parquet file.
        output_path (str): Path to save the final submission CSV.
        debug_sample_batches (int, optional): If set, limits the number of batches processed
                                              for debugging purposes.
    """
    logger = setup_logger("Inference")
    logger.info("Starting inference process...")

    # Initialize components
    feature_gen = IceCubeFeatureGenerator()
    model = GradientBoostingVectorRegressor()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Test Metadata
    logger.info(f"Loading test metadata from {test_meta_path}")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    test_meta = pd.read_parquet(test_meta_path)

    # Group metadata by batch_id for efficient iteration
    # This avoids filtering the large test_meta dataframe inside the loop
    batch_groups = test_meta.groupby("batch_id")

    batch_ids = list(batch_groups.groups.keys())
    logger.info(f"Found {len(batch_ids)} batches to process.")

    if debug_sample_batches is not None:
        logger.info(
            f"DEBUG: Limiting processing to first {debug_sample_batches} batches."
        )
        batch_ids = batch_ids[:debug_sample_batches]

    # Prepare CSV file with header
    with open(output_path, "w") as f:
        f.write("event_id,azimuth,zenith\n")

    total_events_processed = 0

    # Iterate over batches
    for batch_id in batch_ids:
        # Get metadata for this batch
        batch_meta = batch_groups.get_group(batch_id)

        # Determine file path
        # The metadata contains 'batch_file_path'. We take the first one as they are identical for the group.
        rel_path = batch_meta["batch_file_path"].iloc[0]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            logger.warning(f"Batch file not found: {full_path}. Skipping.")
            continue

        # Load batch pulses
        try:
            batch_df = pd.read_parquet(full_path)
        except Exception as e:
            logger.error(f"Failed to read {full_path}: {e}")
            continue

        # Filter pulses to strictly match events in the metadata for this batch
        # This ensures we only predict for the requested events and handles alignment
        valid_events = batch_meta["event_id"].values

        # Ensure batch_df has event_id as a column for filtering
        if "event_id" not in batch_df.columns:
            batch_df = batch_df.reset_index()

        batch_df = batch_df[batch_df["event_id"].isin(valid_events)]

        if batch_df.empty:
            logger.warning(f"No matching events found in batch {batch_id}.")
            continue

        # Compute features
        # We access the internal method to process just this batch without loading everything
        try:
            features = feature_gen._compute_features_for_batch(batch_df)
        except Exception as e:
            logger.error(f"Feature generation failed for batch {batch_id}: {e}")
            continue

        # Predict
        # model.predict returns a DataFrame with index event_id, columns azimuth, zenith
        try:
            preds = model.predict(features)
        except Exception as e:
            logger.error(f"Prediction failed for batch {batch_id}: {e}")
            continue

        # Append to CSV
        # Reset index to get event_id as column
        preds = preds.reset_index()

        # Ensure correct column order and format
        preds = preds[["event_id", "azimuth", "zenith"]]

        # Write to file (append mode)
        preds.to_csv(output_path, mode="a", header=False, index=False)

        count = len(preds)
        total_events_processed += count

        # Cleanup to free memory
        del batch_df, features, preds, batch_meta
        gc.collect()

    logger.info(
        f"Inference completed. Total events processed: {total_events_processed}"
    )
    logger.info(f"Submission saved to {output_path}")
