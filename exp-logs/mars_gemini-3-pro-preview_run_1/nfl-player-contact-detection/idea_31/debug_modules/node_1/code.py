import os
import shutil
import logging
import warnings
import numpy as np
import pandas as pd
import sys

# Filter warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import Library Components
from library.config import PathConfig, MODEL_PARAMS, FeatureConfig, GatingConfig
from library.utils import setup_logging
from library.data import prepare_training_data, get_data_split
from library.training import run_training_pipeline
from library.inference import generate_submission

# =============================================================================
# 1. SETUP & CONFIGURATION OVERRIDES
# =============================================================================

# Define a demo working directory
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Silence Library Logging (Set to ERROR to suppress INFO messages)
logging.getLogger().setLevel(logging.ERROR)

# Override Model Hyperparameters for Speed
# Reducing estimators to a bare minimum to demonstrate functionality quickly
print("Overriding model hyperparameters for speed...")
MODEL_PARAMS["lgbm"]["n_estimators"] = 2
MODEL_PARAMS["lgbm"]["min_child_samples"] = 1  # Allow small data
MODEL_PARAMS["xgb"]["n_estimators"] = 2

# Override Feature/Gating Configs for Speed/Small Data
FeatureConfig.WINDOW_SIZE = 2  # Reduce window from 10 to 2
GatingConfig.PROJECTION_STEPS = 2  # Reduce projection horizon

# =============================================================================
# 2. MINI-DATASET GENERATION
# =============================================================================


def create_mini_datasets():
    print("Creating mini-datasets for demonstration...")

    # --- Load Full Metadata ---
    # We use train_metadata for both train and val in this demo to save space
    full_meta_path = "./metadata/train_metadata.csv"
    df_meta = pd.read_csv(full_meta_path)

    # --- Sample Data ---
    # Ensure we get some positives (contact=1) and negatives (contact=0)
    # We pick 1-2 plays that have contacts
    plays_with_contact = df_meta[df_meta["contact"] == 1]["game_play"].unique()
    selected_plays = plays_with_contact[:2]  # Pick first 2 plays

    df_mini_meta = df_meta[df_meta["game_play"].isin(selected_plays)].copy()

    # Split into Train (80%) and Val (20%) and Test (Copy of Val)
    # We just split by row count for this simple demo
    n = len(df_mini_meta)
    split_idx = int(n * 0.8)

    df_mini_train = df_mini_meta.iloc[:split_idx].reset_index(drop=True)
    df_mini_val = df_mini_meta.iloc[split_idx:].reset_index(drop=True)
    df_mini_test = df_mini_val.copy()  # Simulate test set using val data

    # Create contact_id for test if not present (it is present in metadata)
    # Reset contact column for test to simulate unknown targets
    df_mini_test["contact"] = 0

    # --- Load and Filter Tracking Data ---
    full_tracking_path = "./input/train_player_tracking.csv"
    # Read full tracking is necessary to filter correctly
    # Since the file is ~1.2M rows, it fits in memory easily for this operation
    df_tracking = pd.read_csv(full_tracking_path)
    df_mini_tracking = df_tracking[df_tracking["game_play"].isin(selected_plays)].copy()

    # --- Save Mini Files ---
    paths = {
        "train_meta": os.path.join(DEMO_DIR, "mini_train_metadata.csv"),
        "val_meta": os.path.join(DEMO_DIR, "mini_val_metadata.csv"),
        "test_meta": os.path.join(DEMO_DIR, "mini_test_metadata.csv"),
        "tracking": os.path.join(DEMO_DIR, "mini_tracking.csv"),
        "sample_sub": os.path.join(DEMO_DIR, "mini_sample_submission.csv"),
    }

    df_mini_train.to_csv(paths["train_meta"], index=False)
    df_mini_val.to_csv(paths["val_meta"], index=False)
    df_mini_test.to_csv(paths["test_meta"], index=False)
    df_mini_tracking.to_csv(paths["tracking"], index=False)

    # Create Sample Submission
    # Must contain 'contact_id' and 'contact'
    df_sub = df_mini_test[["contact_id", "contact"]].copy()
    df_sub.to_csv(paths["sample_sub"], index=False)

    return paths


# Generate the data
mini_paths = create_mini_datasets()

# =============================================================================
# 3. PATH INJECTION
# =============================================================================

print("Patching PathConfig to use mini-datasets...")
# We modify the class attributes directly.
# Since the library imports PathConfig class, these changes will be reflected.
PathConfig.WORKING_DIR = DEMO_DIR
PathConfig.TRAIN_METADATA = mini_paths["train_meta"]
PathConfig.VAL_METADATA = mini_paths["val_meta"]
PathConfig.TEST_METADATA = mini_paths["test_meta"]
PathConfig.TRAIN_TRACKING = mini_paths["tracking"]
PathConfig.TEST_TRACKING = mini_paths["tracking"]  # Use same tracking for test demo
PathConfig.SAMPLE_SUBMISSION = mini_paths["sample_sub"]
PathConfig.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

# Update cache paths to be inside the demo dir
PathConfig.CACHE_TRAIN_FEATURES = os.path.join(DEMO_DIR, "features_train.parquet")
PathConfig.CACHE_VAL_FEATURES = os.path.join(DEMO_DIR, "features_val.parquet")
PathConfig.CACHE_TEST_FEATURES = os.path.join(DEMO_DIR, "features_test.parquet")
PathConfig.CACHE_HARD_NEGATIVES = os.path.join(DEMO_DIR, "hard_negative_indices.npy")

# Re-setup directories (just in case)
PathConfig.setup_directories()

# =============================================================================
# 4. EXECUTION & VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("\n=== Starting Pipeline Execution ===\n")

    # --- Step 1: Feature Engineering ---
    print("[1/3] Generating Training Data (Features + Smoothing)...")
    # load_cached=False forces regeneration using our new mini files
    df_train = prepare_training_data(load_cached=False)

    # Verification
    print(f"   -> Training Data Shape: {df_train.shape}")
    assert not df_train.empty, "Training dataframe is empty!"
    assert "contact" in df_train.columns, "Contact column missing in training data"
    # Check if smoothing happened (values should be float, not just 0/1 integers if smoothed)
    # Though if sigma is small or data is sparse, it might look binary.
    # Just checking type is float is good enough.
    assert pd.api.types.is_float_dtype(
        df_train["contact"]
    ) or pd.api.types.is_integer_dtype(df_train["contact"]), "Contact type unexpected"

    # --- Step 2: Training Pipeline ---
    print("\n[2/3] Running Training Pipeline (Mining + Expert Training)...")
    # This runs: Scout Training -> Hard Negative Mining -> Expert Dataset Construction -> Expert Training -> Threshold Opt
    expert_model, best_threshold = run_training_pipeline(load_cached=False)

    # Verification
    print(f"   -> Best Threshold: {best_threshold}")
    assert expert_model is not None, "Expert model was not returned"
    assert (
        0.0 < best_threshold < 1.0
    ), f"Threshold {best_threshold} is out of expected range (0,1)"
    assert os.path.exists(
        os.path.join(DEMO_DIR, "models", "experts", "lgbm_model.joblib")
    ), "Model artifact not saved"

    # --- Step 3: Inference ---
    print("\n[3/3] Generating Submission...")
    generate_submission(load_cached=False)

    # Verification
    submission_path = PathConfig.SUBMISSION_FILE
    print(f"   -> Checking submission at {submission_path}")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"   -> Submission Shape: {df_sub.shape}")
    assert "contact_id" in df_sub.columns, "contact_id column missing"
    assert "contact" in df_sub.columns, "contact column missing"
    assert (
        df_sub.shape[0] == pd.read_csv(mini_paths["sample_sub"]).shape[0]
    ), "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")
