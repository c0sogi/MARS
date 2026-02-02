import os
import sys
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.pipeline import Pipeline


def main():
    print("Initializing NFL Contact Detection Demo...")

    # =========================================================================
    # 1. Configuration & Optimization for Demo Speed
    # =========================================================================
    # Modify Config to run fast:
    # - Use a separate working directory for the demo
    # - Sample only 5000 rows for training to speed up feature engineering
    # - Reduce number of estimators in XGBoost to minimal values

    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Enable Debug Sampling
    Config.DEBUG_SAMPLE_SIZE = 5000

    # Reduce Model Complexity for Speed
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_A["tree_method"] = "hist"  # Fast CPU/GPU histogram

    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_B["tree_method"] = "hist"

    print(f"Configuration updated.")
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"XGB Estimators: {Config.XGB_PARAMS_STREAM_A['n_estimators']}")

    # =========================================================================
    # 2. Instantiate Pipeline
    # =========================================================================
    pipeline = Pipeline()

    # =========================================================================
    # 3. Run Training Pipeline
    # =========================================================================
    print("\n[Step 1/3] Running Training Pipeline...")
    # We set load_cached_data=False to ensure the code actually runs the logic
    # rather than loading pre-existing files from a previous run.
    pipeline.run_training(load_cached_data=False)

    # =========================================================================
    # 4. Verify Training Artifacts
    # =========================================================================
    print("\n[Step 2/3] Verifying Training Results...")

    # Check if models are instantiated
    assert pipeline.model.model_a is not None, "Stream A model failed to train."
    # Note: Stream B might be None if the random sample of 5000 rows contained no Ground contacts,
    # but statistically 5000 rows should contain some. If it is None, we warn but don't fail
    # if it's due to data scarcity in debug mode.
    if pipeline.model.model_b is None:
        print(
            "Warning: Stream B model is None. This may happen if debug sample has no Ground contacts."
        )
    else:
        print("Stream B model trained successfully.")

    # Check if thresholds file exists
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")
    assert os.path.exists(
        thresholds_path
    ), f"Thresholds file missing at {thresholds_path}"

    print("Training verification passed.")

    # =========================================================================
    # 5. Run Inference Pipeline
    # =========================================================================
    print("\n[Step 3/3] Running Inference Pipeline...")
    pipeline.run_inference(load_cached_data=False)

    # =========================================================================
    # 6. Verify Submission
    # =========================================================================
    print("\nVerifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Shape
    print(f"Submission Shape: {df_sub.shape}")

    # Check Columns
    required_cols = {"contact_id", "contact"}
    assert required_cols.issubset(
        df_sub.columns
    ), f"Missing columns. Found: {df_sub.columns}"

    # Check Values
    unique_vals = df_sub["contact"].unique()
    assert np.all(
        np.isin(unique_vals, [0, 1])
    ), f"Invalid values in contact column: {unique_vals}"

    # Check against sample submission length
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length mismatch. Expected {len(df_sample)}, got {len(df_sub)}"

    print("Submission verification passed.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
