import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import the provided library modules
import library.config
import library.utils
import library.feature_engineering
import library.data_factory
import library.training_tabular
import library.training_vision
import library.training_meta


def patch_modules_for_demo():
    """
    Patches the imported modules to use Debug settings and reduced hyperparameters
    for a fast demonstration run. This overrides values imported from config.py.
    """
    print("Patching modules for fast demonstration...")

    # Settings for the demo
    DEMO_DEBUG = True
    DEMO_SAMPLES = 60  # Small number of samples (enough for 5-fold CV)
    DEMO_EPOCHS = 1
    DEMO_LGBM_ESTIMATORS = 20
    DEMO_GLOBAL_MAX_SAMPLES = 10

    # Patch training_tabular
    library.training_tabular.DEBUG = DEMO_DEBUG
    library.training_tabular.MAX_DEBUG_SAMPLES = DEMO_SAMPLES
    # Update dictionary in place
    library.training_tabular.LGBM_PARAMS["n_estimators"] = DEMO_LGBM_ESTIMATORS
    library.training_tabular.LGBM_PARAMS["early_stopping_rounds"] = 5

    # Patch training_vision
    library.training_vision.DEBUG = DEMO_DEBUG
    library.training_vision.MAX_DEBUG_SAMPLES = DEMO_SAMPLES
    library.training_vision.EPOCHS = DEMO_EPOCHS

    # Patch feature_engineering
    library.feature_engineering.GLOBAL_MAX_SAMPLE_SIZE = DEMO_GLOBAL_MAX_SAMPLES

    print("Patching complete.")


def validate_file_exists(filepath, description):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{description} not found at {filepath}")
    print(f"[OK] {description} exists.")


def validate_dataframe_shape(filepath, expected_rows, description):
    df = pd.read_csv(filepath)
    if len(df) != expected_rows:
        raise AssertionError(
            f"{description} has {len(df)} rows, expected {expected_rows}."
        )
    print(f"[OK] {description} has correct shape ({len(df)} rows).")


def main():
    # 1. Setup
    library.utils.seed_everything(42)
    patch_modules_for_demo()

    # Ensure working directory exists (as defined in config)
    work_dir = library.config.WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    print("\n" + "=" * 40)
    print("1. Running Tabular Branch (LightGBM)")
    print("=" * 40)

    # Run Tabular CV
    # We set load_cached_data=False to force feature generation for the demo subset
    oof_tab, test_tab = library.training_tabular.run_lgbm_cv(load_cached_data=False)

    # Validate Tabular Outputs
    lgbm_oof_path = os.path.join(work_dir, "lgbm_oof.csv")
    lgbm_test_path = os.path.join(work_dir, "lgbm_test.csv")

    validate_file_exists(lgbm_oof_path, "Tabular OOF Predictions")
    validate_file_exists(lgbm_test_path, "Tabular Test Predictions")

    # In debug mode, we expect roughly MAX_DEBUG_SAMPLES rows
    # (Exact number depends on availability in the source files, but should match the DF returned)
    validate_dataframe_shape(lgbm_oof_path, len(oof_tab), "Tabular OOF CSV")

    print("\n" + "=" * 40)
    print("2. Running Vision Branch (EfficientNet)")
    print("=" * 40)

    # Run Vision CV
    # This handles spectrogram generation internally
    oof_vis, test_vis = library.training_vision.run_vision_cv(load_cached_data=False)

    # Validate Vision Outputs
    cnn_oof_path = os.path.join(work_dir, "cnn_oof.csv")
    cnn_test_path = os.path.join(work_dir, "cnn_test.csv")

    validate_file_exists(cnn_oof_path, "Vision OOF Predictions")
    validate_file_exists(cnn_test_path, "Vision Test Predictions")
    validate_dataframe_shape(cnn_oof_path, len(oof_vis), "Vision OOF CSV")

    print("\n" + "=" * 40)
    print("3. Running Meta-Learner (Stacking)")
    print("=" * 40)

    # Run Stacking
    submission_df = library.training_meta.train_ridge_stack(
        oof_tabular=oof_tab,
        test_tabular=test_tab,
        oof_vision=oof_vis,
        test_vision=test_vis,
    )

    # Validate Submission
    submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")
    validate_file_exists(submission_path, "Submission File")

    # Check submission format
    if list(submission_df.columns) != ["segment_id", "time_to_eruption"]:
        raise AssertionError("Submission columns do not match requirements.")

    # Check for no NaNs
    if submission_df.isnull().any().any():
        raise AssertionError("Submission contains NaN values.")

    print(f"[OK] Submission generated successfully with {len(submission_df)} rows.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
