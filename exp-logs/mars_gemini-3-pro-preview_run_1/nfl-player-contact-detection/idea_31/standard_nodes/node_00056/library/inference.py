import os
import numpy as np
import pandas as pd
import logging
from library.config import PathConfig
from library.data import get_data_split
from library.models import TriEnsemble
from library.utils import setup_logging, load_numpy

# Initialize logging
setup_logging()

# Columns to exclude from features (must match training configuration)
META_COLS = [
    "contact_id",
    "game_play",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
    "datetime",
]


def get_feature_cols(df):
    """Returns the list of feature columns by excluding metadata."""
    return [c for c in df.columns if c not in META_COLS]


def generate_submission(load_cached=True):
    """
    Executes the inference pipeline:
    1. Loads processed test data (features).
    2. Loads the trained Tri-Ensemble model.
    3. Loads the optimized threshold.
    4. Generates predictions.
    5. Saves the submission file.

    Args:
        load_cached (bool): Whether to use cached feature data.
    """
    logging.info("Starting Submission Generation...")

    # 1. Load Test Data
    # get_data_split("test") triggers FeatureEngineer(mode="test")
    # which ensures NO gating is applied, preserving all rows from sample_submission.
    df_test = get_data_split("test", load_cached=load_cached)

    # Identify feature columns
    feature_cols = get_feature_cols(df_test)
    logging.info(f"Inference Feature Columns ({len(feature_cols)}): {feature_cols}")

    X_test = df_test[feature_cols]

    # 2. Load Model
    model_dir = os.path.join(PathConfig.WORKING_DIR, "models", "experts")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at {model_dir}. Please run training first."
        )

    logging.info(f"Loading Tri-Ensemble models from {model_dir}...")
    ensemble = TriEnsemble()
    ensemble.load(model_dir)

    # 3. Load Threshold
    thresh_path = os.path.join(PathConfig.WORKING_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_thresh = load_numpy(thresh_path)[0]
        logging.info(f"Loaded optimized threshold: {best_thresh}")
    else:
        best_thresh = 0.5
        logging.warning(
            f"Threshold file not found at {thresh_path}. Defaulting to 0.5."
        )

    # 4. Generate Predictions
    logging.info(f"Predicting on {len(X_test)} test samples...")
    probs = ensemble.predict(X_test)

    # Apply Threshold
    preds_bin = (probs >= best_thresh).astype(int)

    # 5. Format Submission
    # We need to map predictions back to contact_ids.
    # df_test contains 'contact_id' from the metadata processing.

    submission_df = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": preds_bin}
    )

    # Ensure we match the sample submission order/completeness
    # Load sample submission to verify
    sample_sub = pd.read_csv(PathConfig.SAMPLE_SUBMISSION)

    # Merge with sample submission to ensure correct order and row count
    # Left join on sample submission ensures we have all required rows
    final_submission = sample_sub[["contact_id"]].merge(
        submission_df, on="contact_id", how="left"
    )

    # Fill missing (if any, though shouldn't be with correct pipeline) with 0
    missing_count = final_submission["contact"].isna().sum()
    if missing_count > 0:
        logging.warning(
            f"{missing_count} rows were missing predictions. Filling with 0."
        )
        final_submission["contact"] = final_submission["contact"].fillna(0).astype(int)

    # Save
    save_path = PathConfig.SUBMISSION_FILE
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_submission.to_csv(save_path, index=False)

    logging.info(f"Submission saved to {save_path}")
    logging.info(
        f"Predicted Contact Count: {final_submission['contact'].sum()} / {len(final_submission)}"
    )
