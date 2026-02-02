import os
import shutil
import pandas as pd
import numpy as np
import sys

# Import from the provided library
from library.config import Config
from library.workflow_manager import WorkflowManager
from library.utils import set_seed


def setup_demo_config():
    """
    Overrides default configuration for a fast, verifiable demo run.
    """
    print("Setting up demo configuration...")

    # Enable Debug mode to subsample data (processes only ~2 games)
    Config.DEBUG = True

    # Reduce boosting rounds for speed
    Config.NUM_BOOST_ROUND = 10
    Config.EARLY_STOPPING_ROUNDS = 5

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Verify paths
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Submission Path: {Config.SUBMISSION_PATH}")


def validate_stream_a_results(model, threshold, oof_df, val_preds):
    """
    Validates the outputs of the Stream A training process.
    """
    print("Validating Stream A results...")

    # Check Model
    if model is None or model.model is None:
        raise AssertionError("Stream A model was not trained successfully.")

    # Check Threshold
    if not (0.0 < threshold < 1.0):
        raise AssertionError(
            f"Stream A threshold {threshold} is invalid (expected 0-1)."
        )

    # Check OOF Predictions (Context for Stream B)
    if oof_df is None or oof_df.empty:
        raise AssertionError("Stream A OOF predictions are empty.")

    required_cols = ["contact_id", "prob"]
    for col in required_cols:
        if col not in oof_df.columns:
            raise AssertionError(f"Stream A OOF predictions missing column: {col}")

    print(
        f"Stream A Validation Passed. Threshold: {threshold:.4f}, OOF Shape: {oof_df.shape}"
    )


def validate_stream_b_results(model, threshold):
    """
    Validates the outputs of the Stream B training process.
    """
    print("Validating Stream B results...")

    if model is None or model.model is None:
        raise AssertionError("Stream B model was not trained successfully.")

    if not (0.0 < threshold < 1.0):
        raise AssertionError(
            f"Stream B threshold {threshold} is invalid (expected 0-1)."
        )

    print(f"Stream B Validation Passed. Threshold: {threshold:.4f}")


def validate_submission():
    """
    Validates the final submission file.
    """
    print("Validating submission file...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    if "contact_id" not in df_sub.columns or "contact" not in df_sub.columns:
        raise AssertionError(
            "Submission file missing required columns ('contact_id', 'contact')."
        )

    # Check values
    if not df_sub["contact"].isin([0, 1]).all():
        raise AssertionError("Submission 'contact' column contains non-binary values.")

    # Check if empty
    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"Submission Validation Passed. Rows: {len(df_sub)}")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()
    set_seed(Config.SEED)

    # Initialize Manager
    wm = WorkflowManager()

    # 2. Train Stream A (Interaction)
    # We set load_cached_data=False to ensure the code actually runs through the logic
    print("\n--- Step 1: Training Stream A (Interaction) ---")
    model_a, thresh_a, train_oof_a, val_preds_a = wm.train_interaction_stream(
        load_cached_data=False
    )

    validate_stream_a_results(model_a, thresh_a, train_oof_a, val_preds_a)

    # 3. Train Stream B (Impact)
    # Uses Stream A OOF predictions as input features
    print("\n--- Step 2: Training Stream B (Impact) ---")
    model_b, thresh_b = wm.train_impact_stream(
        train_oof_a, val_preds_a, load_cached_data=False
    )

    validate_stream_b_results(model_b, thresh_b)

    # 4. Run Inference
    print("\n--- Step 3: Running Inference Cascade ---")
    wm.run_inference_cascade(
        model_a, thresh_a, model_b, thresh_b, load_cached_data=False
    )

    # 5. Final Validation
    validate_submission()

    print("\nDemo execution completed successfully.")
