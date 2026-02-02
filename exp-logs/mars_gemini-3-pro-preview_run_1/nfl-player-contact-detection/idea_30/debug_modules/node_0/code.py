import os
import sys
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import Config
from library.workflow import Workflow
from library.utils import seed_everything


def run_demo():
    print("Initializing OSVA-E Pipeline Demo...")

    # =========================================================================
    # 1. Configuration Overrides for Speed & Demo Purposes
    # =========================================================================
    # We modify the Config class attributes directly to optimize for a fast demo run.

    # Enable Debug mode to sample a small subset of data (e.g., 2000 rows)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000

    # Redirect working directory to a demo folder
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.SUBMISSION_DIR = Config.WORKING_DIR  # Save submission in demo dir
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update paths derived from the new working directory
    Config.PROCESSED_TRAIN_PATH = os.path.join(
        Config.CACHE_DIR, "features_train_full.parquet"
    )
    Config.PROCESSED_VAL_PATH = os.path.join(
        Config.CACHE_DIR, "features_val_full.parquet"
    )
    Config.PROCESSED_TEST_PATH = os.path.join(
        Config.CACHE_DIR, "features_test_full.parquet"
    )
    Config.HARD_NEGATIVE_INDICES_PATH = os.path.join(
        Config.CACHE_DIR, "hard_negative_indices.npy"
    )

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce Model Complexity for fast training
    # LightGBM
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    # XGBoost
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    # Note: LGBMDartModel inherits LGBM_PARAMS but adds DART specific params.
    # The reduction in n_estimators above applies to it as well.

    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG} (Sample Size: {Config.DEBUG_SAMPLE_SIZE})")
    print(f"Estimators per model: {Config.LGBM_PARAMS['n_estimators']}")

    # =========================================================================
    # 2. Pipeline Execution
    # =========================================================================
    # Instantiate the workflow
    workflow = Workflow()

    # Run the full pipeline
    # load_cached_data=False ensures we actually run the feature engineering logic
    print("\nStarting Workflow Execution...")
    workflow.run_full_workflow(load_cached_data=False)

    # =========================================================================
    # 3. Verification & Assertions
    # =========================================================================
    print("\nVerifying Outputs...")

    # A. Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Verify row count matches sample submission
    assert len(df_sub) == len(
        df_sample
    ), f"Submission row count mismatch! Expected {len(df_sample)}, got {len(df_sub)}"

    # Verify columns
    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission file missing required columns."

    print(f"[PASS] Submission file created with {len(df_sub)} rows.")

    # B. Check Model Artifacts
    scout_dir = os.path.join(Config.MODEL_DIR, "scouts")
    expert_dir = os.path.join(Config.MODEL_DIR, "experts")

    expected_models = ["lgbm_model.joblib", "xgb_model.joblib", "dart_model.joblib"]

    # Verify Scouts
    for m in expected_models:
        path = os.path.join(scout_dir, m)
        assert os.path.exists(path), f"Missing Scout model: {m}"
    print("[PASS] Scout models saved.")

    # Verify Experts
    for m in expected_models:
        path = os.path.join(expert_dir, m)
        assert os.path.exists(path), f"Missing Expert model: {m}"
    print("[PASS] Expert models saved.")

    # C. Check Hard Negative Mining
    if not os.path.exists(Config.HARD_NEGATIVE_INDICES_PATH):
        raise FileNotFoundError("Hard negative indices file not found.")

    hard_negs = np.load(Config.HARD_NEGATIVE_INDICES_PATH)
    # Note: In a very small random sample, we might not find hard negatives if the model is perfect
    # or if no negatives cross the threshold. However, usually some will be found.
    # We just check the file is readable and is an array.
    assert isinstance(
        hard_negs, np.ndarray
    ), "Hard negative indices is not a numpy array."
    print(f"[PASS] Hard negative mining produced {len(hard_negs)} indices.")

    # D. Check Threshold
    thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
    assert os.path.exists(thresh_path), "Best threshold file not found."
    thresh = np.load(thresh_path)
    print(f"[PASS] Optimized Threshold: {thresh[0]}")

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
