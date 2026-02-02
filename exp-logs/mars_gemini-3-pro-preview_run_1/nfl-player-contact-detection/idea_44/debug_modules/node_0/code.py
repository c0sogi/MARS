import os
import pandas as pd
import numpy as np
import logging
import sys

# Import library components
from library.config import Config
from library.training_curriculum import CurriculumManager
from library.inference import InferenceEngine
from library.data_processing import DataLoader


# =============================================================================
# 1. Configuration & Optimization for Demo
# =============================================================================
def configure_fast_run():
    print("Configuring environment for fast demonstration...")

    # Update Config parameters to run minimally
    Config.EXP_NAME = "demo_verification_run"
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.SUBMISSION_DIR = "./working/submission_demo"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set Debug mode
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Process very few samples

    # Minimize Model Complexity for Speed
    Config.LGBM_PARAMS["n_estimators"] = 5
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["verbose"] = -1

    Config.XGB_PARAMS["n_estimators"] = 5
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    Config.XGB_PARAMS["verbosity"] = 0

    # Adjust Physics/Mining parameters to ensure we get *some* data through gating
    Config.GATING_DIST = (
        100.0  # Relax gating to ensure small sample isn't filtered out entirely
    )
    Config.HARD_NEGATIVE_RATIO = 1.0
    Config.ANCHOR_RATIO = 1.0


# =============================================================================
# 2. Monkey Patching DataLoader for Speed
# =============================================================================
# The datasets are massive (millions of rows). For this demo, we intercept
# the loading methods to return small slices. This simulates the pipeline
# logic without the I/O overhead.

original_load_meta = DataLoader._load_raw_metadata
original_load_track = DataLoader._load_raw_tracking


def fast_load_raw_metadata(self, split: str) -> pd.DataFrame:
    # Map split to path
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"[Patch] Loading head of metadata from {path}")
    # Load enough rows to form a valid batch
    return pd.read_csv(path, nrows=500)


def fast_load_raw_tracking(self, split: str) -> pd.DataFrame:
    if split in ["train", "val"]:
        path = Config.TRAIN_TRACKING_PATH
    elif split == "test":
        path = Config.TEST_TRACKING_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"[Patch] Loading head of tracking from {path}")
    # Load enough tracking data to likely overlap with the metadata head
    # Tracking is usually sorted by game/play/step similar to metadata
    return pd.read_csv(path, nrows=10000)


# Apply Patch
DataLoader._load_raw_metadata = fast_load_raw_metadata
DataLoader._load_raw_tracking = fast_load_raw_tracking


# =============================================================================
# 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Setup
    configure_fast_run()

    # --- Step 1: Training Curriculum ---
    print("\n" + "=" * 40)
    print("Step 1: Executing Training Curriculum")
    print("=" * 40)

    # Instantiate Manager
    trainer = CurriculumManager()

    # Run Training (Force reload to use our patched data loaders)
    # This will:
    # 1. Load (patched) data
    # 2. Compute features
    # 3. Train Scouts -> Mine Hard Negatives -> Train Experts
    lgbm_model, xgb_model = trainer.train(load_cached_data=False)

    # Verification: Check if models are saved
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    expected_models = ["expert_lgbm.joblib", "expert_xgb.joblib"]
    for m in expected_models:
        path = os.path.join(model_dir, m)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model training failed: {path} not created.")
        print(f"Verified model artifact: {path}")

    # --- Step 2: Inference Pipeline ---
    print("\n" + "=" * 40)
    print("Step 2: Executing Inference Pipeline")
    print("=" * 40)

    # Instantiate Engine
    inference = InferenceEngine()

    # Run Inference
    # This will:
    # 1. Load saved models
    # 2. Optimize threshold on (patched) val set
    # 3. Predict on (patched) test set
    # 4. Generate submission csv
    inference.run_inference(load_cached_data=False)

    # Verification: Check submission
    sub_path = Config.SUBMISSION_FILE
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Inference failed: {sub_path} not created.")

    df_sub = pd.read_csv(sub_path)
    print(f"Submission generated at: {sub_path}")
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Logic Checks
    required_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(f"Submission missing columns. Found: {df_sub.columns}")

    if not df_sub["contact"].isin([0, 1]).all():
        raise AssertionError(
            "Submission contains non-binary values in 'contact' column."
        )

    print("\nSUCCESS: Pipeline demonstration completed and verified.")
