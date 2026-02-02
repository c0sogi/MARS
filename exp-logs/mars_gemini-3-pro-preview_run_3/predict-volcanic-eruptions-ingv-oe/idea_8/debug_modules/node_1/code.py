import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import shutil

# Import library modules
import library.config as config
from library.signal_processing import impute_missing_values, apply_savgol_filter
from library.feature_engineering import process_segment
from library.data_manager import build_dataset
from library.stacking_trainer import StackingEnsemble, generate_submission


# ==========================================
# Setup & Configuration Overrides
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def optimize_config_for_demo():
    """
    Monkey-patch the configuration to run a fast demo.
    """
    print("Optimizing configuration for fast demonstration...")

    # Reduce CV folds
    config.N_FOLDS = 2

    # Reduce model complexity/iterations
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8

    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 3

    config.HGB_PARAMS["max_iter"] = 10
    config.HGB_PARAMS["max_leaf_nodes"] = 8

    # Ensure working directory is clean for this run if needed,
    # but we'll just let the library handle directory creation.
    print(f"Config N_FOLDS set to: {config.N_FOLDS}")


# ==========================================
# Verification Functions
# ==========================================


def verify_signal_processing():
    print("\n=== Verifying Signal Processing ===")

    # Load a sample file path from metadata
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    sample_row = train_meta.iloc[0]
    file_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    # Load Raw Data
    df = pd.read_csv(file_path, dtype="float32")
    original_shape = df.shape
    print(f"Loaded sample {sample_row['segment_id']} with shape {original_shape}")

    # 1. Test Imputation
    # Introduce artificial NaNs to test imputation
    df_nan = df.copy()
    df_nan.iloc[0, 0] = np.nan
    df_imputed = impute_missing_values(df_nan)

    assert not df_imputed.isnull().values.any(), "Imputation failed: NaNs remain."
    assert df_imputed.shape == original_shape, "Imputation changed DataFrame shape."
    print("Imputation logic verified.")

    # 2. Test Savitzky-Golay Filter
    df_smoothed = apply_savgol_filter(df_imputed)
    assert df_smoothed.shape == original_shape, "Smoothing changed DataFrame shape."
    assert not df_smoothed.isnull().values.any(), "Smoothing introduced NaNs."

    # Check that values actually changed (smoothing happened)
    # It's possible for very smooth signals to remain identical, but unlikely with sensor noise.
    if not np.allclose(df_imputed.values, df_smoothed.values):
        print("Savitzky-Golay filter verified (signal modified).")
    else:
        print("Warning: Smoothed signal is identical to input (check filter params).")


def verify_feature_engineering():
    print("\n=== Verifying Feature Engineering ===")

    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    sample_row = train_meta.iloc[0]
    file_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    # Test process_segment
    features = process_segment(file_path)

    assert isinstance(features, dict), "process_segment should return a dictionary."
    assert len(features) > 0, "Feature dictionary is empty."

    # Check for specific feature groups
    keys = features.keys()
    has_kinematics = any("vel_mean" in k for k in keys)
    has_freq = any("spec_centroid" in k for k in keys)
    has_spatial = any("corr_" in k for k in keys)

    assert has_kinematics, "Kinematic features missing."
    assert has_freq, "Frequency features missing."
    assert has_spatial, "Spatial interaction features missing."

    print(f"Feature extraction successful. Generated {len(features)} features.")


def verify_pipeline_execution():
    print("\n=== Verifying Full Pipeline (Data Load -> Train -> Predict) ===")

    # 1. Build Datasets (Debug Mode)
    debug_train_size = 50
    debug_test_size = 20

    print(f"Building training set (n={debug_train_size})...")
    X_train, y_train, train_ids = build_dataset("train", debug_size=debug_train_size)

    assert (
        len(X_train) == debug_train_size
    ), f"Expected {debug_train_size} training samples, got {len(X_train)}"
    assert len(y_train) == debug_train_size, "Mismatch between X and y lengths."
    assert not X_train.isnull().values.any(), "Training features contain NaNs."

    print(f"Building test set (n={debug_test_size})...")
    X_test, test_ids = build_dataset("test", debug_size=debug_test_size)

    assert (
        len(X_test) == debug_test_size
    ), f"Expected {debug_test_size} test samples, got {len(X_test)}"

    # 2. Train Stacking Ensemble
    print("Initializing Stacking Ensemble...")
    ensemble = StackingEnsemble()

    print("Fitting Ensemble (Level 0 + Level 1 + Retraining)...")
    ensemble.fit(X_train, y_train)

    # Verify models are stored
    assert ensemble.trained_meta_model is not None, "Meta learner was not trained."
    assert len(ensemble.trained_base_models) == 3, "Not all base models were retrained."

    # 3. Predict and Generate Submission
    print("Generating predictions on test set...")
    preds = ensemble.predict(X_test)

    assert len(preds) == debug_test_size, "Prediction length mismatch."
    assert not np.isnan(preds).any(), "Predictions contain NaNs."

    # 4. Save Submission
    # We override the submission path to not overwrite the main one if it exists
    original_sub_path = config.SUBMISSION_PATH
    demo_sub_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    # Temporarily patch the path in the module if needed, or just pass to generate_submission if it supported it.
    # The generate_submission function uses config.SUBMISSION_PATH directly.
    # We will temporarily change the config variable.
    config.SUBMISSION_PATH = demo_sub_path

    generate_submission(ensemble, X_test, test_ids)

    assert os.path.exists(demo_sub_path), "Submission file was not created."

    # Verify file content
    sub_df = pd.read_csv(demo_sub_path)
    assert sub_df.shape == (debug_test_size, 2), "Submission file has incorrect shape."
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns incorrect."

    print(f"Pipeline verification successful. Submission saved to {demo_sub_path}")


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    try:
        print("Starting Code Demonstration...")
        set_seed(42)

        # 1. Optimize Config
        optimize_config_for_demo()

        # 2. Verify Signal Processing
        verify_signal_processing()

        # 3. Verify Feature Engineering
        verify_feature_engineering()

        # 4. Verify Full Pipeline
        verify_pipeline_execution()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAILURE] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILURE] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
