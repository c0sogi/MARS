import os
import sys
import pandas as pd
import numpy as np
import torch
import logging

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.train_lgbm import run_lgbm_cv
from library.train_cnn import run_cnn_cv
from library.meta_learner import train_ridge_stacker


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Volcano Eruption Prediction Demo...")

    # Override Config for a fast demonstration (Debug Mode)
    # This ensures we process only a small subset of data (50 files)
    Config.DEBUG = True
    Config.N_FOLDS = 2  # Use 2 folds for speed
    Config.WORKING_DIR = (
        "./working/demo_execution"  # Separate working dir to avoid cache conflicts
    )

    # LightGBM Speed Optimizations
    Config.LGB_PARAMS["n_estimators"] = 20
    Config.LGB_PARAMS["early_stopping_rounds"] = 5

    # CNN Speed Optimizations
    Config.CNN_PARAMS["epochs"] = 1
    Config.CNN_PARAMS["batch_size"] = 8

    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Configuration: DEBUG={Config.DEBUG}, Folds={Config.N_FOLDS}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Branch A: LightGBM (Tabular)
    # ==========================================
    print("\n" + "=" * 40)
    print("Step 1: Running LightGBM Pipeline")
    print("=" * 40)

    try:
        # run_lgbm_cv handles feature extraction, training, and prediction
        lgb_oof, lgb_test = run_lgbm_cv(debug=True)

        # Validation
        assert isinstance(lgb_oof, pd.DataFrame)
        assert isinstance(lgb_test, pd.DataFrame)
        assert not lgb_oof.empty, "LightGBM OOF DataFrame is empty."
        assert not lgb_test.empty, "LightGBM Test DataFrame is empty."
        assert "pred" in lgb_oof.columns, "OOF missing 'pred' column."

        print(
            f"LightGBM Success. OOF Shape: {lgb_oof.shape}, Test Shape: {lgb_test.shape}"
        )

    except Exception as e:
        print(f"LightGBM Pipeline Failed: {e}")
        raise e

    # ==========================================
    # 3. Branch B: CNN (Vision)
    # ==========================================
    print("\n" + "=" * 40)
    print("Step 2: Running CNN Pipeline")
    print("=" * 40)

    try:
        # run_cnn_cv handles spectrogram generation, CNN training, and prediction
        cnn_oof, cnn_test = run_cnn_cv(debug=True)

        # Validation
        assert isinstance(cnn_oof, pd.DataFrame)
        assert isinstance(cnn_test, pd.DataFrame)
        assert not cnn_oof.empty, "CNN OOF DataFrame is empty."
        assert not cnn_test.empty, "CNN Test DataFrame is empty."
        assert "pred" in cnn_oof.columns, "OOF missing 'pred' column."

        print(f"CNN Success. OOF Shape: {cnn_oof.shape}, Test Shape: {cnn_test.shape}")

    except Exception as e:
        print(f"CNN Pipeline Failed: {e}")
        raise e

    # ==========================================
    # 4. Meta-Learner (Stacking)
    # ==========================================
    print("\n" + "=" * 40)
    print("Step 3: Running Meta-Learner Stacking")
    print("=" * 40)

    try:
        # Combine predictions using Ridge Regression
        submission, meta_model = train_ridge_stacker(
            lgb_oof_df=lgb_oof,
            lgb_test_df=lgb_test,
            cnn_oof_df=cnn_oof,
            cnn_test_df=cnn_test,
        )

        # Validation
        assert isinstance(submission, pd.DataFrame)
        assert "segment_id" in submission.columns
        assert "time_to_eruption" in submission.columns
        assert not submission.empty, "Submission DataFrame is empty."
        # Ensure predictions are non-negative (physical constraint)
        assert (
            submission["time_to_eruption"] >= 0
        ).all(), "Found negative predictions."

        print(f"Stacking Success. Submission Shape: {submission.shape}")
        print("Sample Predictions:")
        print(submission.head())

    except Exception as e:
        print(f"Meta-Learner Failed: {e}")
        raise e

    print("\n" + "=" * 40)
    print("Demonstration Completed Successfully")
    print("=" * 40)


if __name__ == "__main__":
    main()
