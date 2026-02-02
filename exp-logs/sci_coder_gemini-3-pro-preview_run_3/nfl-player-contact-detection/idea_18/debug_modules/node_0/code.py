import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from library.config import Config
from library.utils import set_seed
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.model_engine import DualStreamGBDT

# =============================================================================
# 1. Setup and Configuration Overrides for Demo Speed
# =============================================================================


def configure_demo_settings():
    """
    Overrides default configuration to ensure the demo runs quickly.
    Reduces number of estimators and depth for XGBoost.
    """
    print("[Demo] Configuring lightweight model settings...")

    # Reduce computational load for Stream A
    Config.STREAM_A_PARAMS["n_estimators"] = 10
    Config.STREAM_A_PARAMS["max_depth"] = 3
    Config.STREAM_A_PARAMS["tree_method"] = (
        "hist"  # Use CPU compatible for guaranteed run if GPU busy/small
    )
    if "predictor" in Config.STREAM_A_PARAMS:
        del Config.STREAM_A_PARAMS["predictor"]  # Let XGBoost decide

    # Reduce computational load for Stream B
    Config.STREAM_B_PARAMS["n_estimators"] = 10
    Config.STREAM_B_PARAMS["max_depth"] = 3
    Config.STREAM_B_PARAMS["tree_method"] = "hist"
    if "predictor" in Config.STREAM_B_PARAMS:
        del Config.STREAM_B_PARAMS["predictor"]

    # Reduce early stopping rounds
    Config.EARLY_STOPPING_ROUNDS = 5

    # Use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)


# =============================================================================
# 2. Monkey Patching for Data Subsampling
# =============================================================================

# Store original method to call it internally if needed, or just reimplement logic
_original_load_metadata = DataLoader.load_metadata


def fast_load_metadata(self, mode):
    """
    Monkey-patched version of load_metadata that returns a subset of plays.
    This preserves temporal structure (required for lags) while reducing size.
    """
    print(f"[Demo] Loading and subsampling metadata for mode: {mode}")

    # Load full file using pandas directly (bypassing the original method to keep it simple)
    if mode == "train":
        path = self.config.TRAIN_META_PATH
    elif mode == "validation":
        path = self.config.VAL_META_PATH
    elif mode == "test":
        path = self.config.TEST_META_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    df = pd.read_csv(path)

    # Subsample: Keep only the first 3 unique game_plays
    # This ensures we have enough data for lags but not too much to process
    unique_plays = df["game_play"].unique()
    if len(unique_plays) > 3:
        selected_plays = unique_plays[:3]
        df_subset = df[df["game_play"].isin(selected_plays)].copy()
        print(
            f"[Demo] Subsampled {mode} from {len(unique_plays)} to {len(selected_plays)} plays."
        )
        return df_subset

    return df


# Apply the patch
DataLoader.load_metadata = fast_load_metadata

# =============================================================================
# 3. Execution Pipeline
# =============================================================================


def run_demo():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Apply settings
    configure_demo_settings()

    print("\n" + "=" * 40)
    print("STEP 1: Feature Engineering")
    print("=" * 40)

    # Instantiate Feature Engineer
    fe = FeatureEngineer()

    # Generate Train Features
    # Note: load_cached_data=False forces generation to test the logic
    print("Generating Training Features...")
    (X_a_train, y_a_train, ids_a_train), (X_b_train, y_b_train, ids_b_train) = (
        fe.generate_features("train", load_cached_data=False)
    )

    # Validation: Check shapes
    assert isinstance(X_a_train, pd.DataFrame), "Stream A X must be DataFrame"
    assert isinstance(y_a_train, np.ndarray), "Stream A y must be numpy array"
    assert len(X_a_train) == len(y_a_train), "Mismatch in X and y lengths for Stream A"

    # Validation: Check for specific engineered features
    expected_cols_a = ["dist_p1_p2_lag0", "rel_surge_lag0"]
    for col in expected_cols_a:
        assert col in X_a_train.columns, f"Missing expected feature {col} in Stream A"

    print(f"Stream A Train Shape: {X_a_train.shape}")
    print(f"Stream B Train Shape: {X_b_train.shape}")

    # Generate Validation Features
    print("\nGenerating Validation Features...")
    (X_a_val, y_a_val, ids_a_val), (X_b_val, y_b_val, ids_b_val) = fe.generate_features(
        "validation", load_cached_data=False
    )

    assert (
        X_a_val.shape[1] == X_a_train.shape[1]
    ), "Feature mismatch between Train and Val"

    print("\n" + "=" * 40)
    print("STEP 2: Model Training")
    print("=" * 40)

    # Instantiate Engine
    engine = DualStreamGBDT()

    # Train
    # We pass the tuples generated by FeatureEngineer directly
    engine.train(
        train_data_a=(X_a_train, y_a_train, ids_a_train),
        val_data_a=(X_a_val, y_a_val, ids_a_val),
        train_data_b=(X_b_train, y_b_train, ids_b_train),
        val_data_b=(X_b_val, y_b_val, ids_b_val),
    )

    # Validation: Check if models exist
    assert engine.model_a is not None, "Model A failed to train"
    assert engine.model_b is not None, "Model B failed to train"

    # Validation: Check if thresholds are reasonable (between 0 and 1)
    assert 0.0 < engine.threshold_a < 1.0, f"Invalid threshold A: {engine.threshold_a}"
    assert 0.0 < engine.threshold_b < 1.0, f"Invalid threshold B: {engine.threshold_b}"

    print("Training complete. Models and metadata saved.")

    print("\n" + "=" * 40)
    print("STEP 3: Inference and Submission")
    print("=" * 40)

    # Generate Test Features
    print("Generating Test Features...")
    (X_a_test, y_a_test, ids_a_test), (X_b_test, y_b_test, ids_b_test) = (
        fe.generate_features("test", load_cached_data=False)
    )

    # Generate Submission
    engine.generate_submission(
        test_data_a=(X_a_test, y_a_test, ids_a_test),
        test_data_b=(X_b_test, y_b_test, ids_b_test),
    )

    # Validation: Check Output File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Validation: Check columns
    assert "contact_id" in df_sub.columns, "Missing contact_id in submission"
    assert "contact" in df_sub.columns, "Missing contact in submission"

    # Validation: Check values
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    # Validation: Check against sample submission length
    # Note: In a real run, this should match exactly.
    # Since we didn't filter the sample_submission.csv file itself (only the metadata loading),
    # the engine.generate_submission method loads the full sample_submission and merges our predictions.
    # Rows not in our test subset (because we subsampled test metadata) will be filled with 0.
    # So the length should match the original sample submission.
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length mismatch. Expected {len(df_sample)}, got {len(df_sub)}"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
