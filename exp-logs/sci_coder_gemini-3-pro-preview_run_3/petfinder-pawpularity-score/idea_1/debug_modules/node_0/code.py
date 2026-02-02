import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_rmse
from library.data_loader import get_dataloaders
from library.feature_extractor import extract_features
from library.regressor import RidgeRegressor, train_and_evaluate


def main():
    print("==================================================")
    print("   Pawpularity Prediction Pipeline Demonstration  ")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Optimize for speed: Use Debug mode with a small subset of data
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images
    Config.BATCH_SIZE = 8  # Small batch size for the subset
    Config.RIDGE_ALPHA = 1.0  # Regularization strength

    # Ensure reproducibility
    set_seed(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch a single batch to verify shapes and types
    try:
        images, meta, targets, ids = next(iter(train_loader))

        print(
            f"  Batch Shapes -> Images: {images.shape}, Meta: {meta.shape}, Targets: {targets.shape}"
        )

        # Assertions to verify logic
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Image tensor shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}"
        assert meta.shape == (
            Config.BATCH_SIZE,
            12,
        ), "Metadata tensor shape mismatch. Expected (Batch, 12)"
        assert targets.shape == (
            Config.BATCH_SIZE,
        ), "Target tensor shape mismatch. Expected (Batch,)"
        assert (
            isinstance(ids, tuple)
            or isinstance(ids, list)
            or isinstance(ids, torch.Tensor)
        ), "IDs should be a list/tuple/tensor"

        print("  Data Loading verification passed.")

    except StopIteration:
        raise RuntimeError(
            "DataLoader is empty. Check DEBUG_SUBSET_SIZE vs BATCH_SIZE."
        )

    # ---------------------------------------------------------
    # 3. Feature Extraction Demonstration
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Feature Extraction (MobileNetV2)...")
    # load_cached_data=False forces the code to run the CNN inference instead of loading .npy files
    # This demonstrates the feature_extractor logic.
    data = extract_features(load_cached_data=False)

    train_img, train_meta, train_y = data["train"]
    val_img, val_meta, val_y = data["val"]
    test_img, test_meta, test_ids = data["test"]

    print(f"  Extracted Train Image Features: {train_img.shape}")
    print(f"  Extracted Train Meta Features:  {train_meta.shape}")

    # Assertions
    # MobileNetV2 (headless) outputs 1280-dimensional vectors
    assert train_img.shape[1] == 1280, "Image feature dimension should be 1280"
    assert train_meta.shape[1] == 12, "Metadata feature dimension should be 12"
    assert (
        train_img.shape[0] == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} training samples in debug mode"

    print("  Feature Extraction verification passed.")

    # ---------------------------------------------------------
    # 4. Regressor Training & Prediction
    # ---------------------------------------------------------
    print("\n[4] Demonstrating Ridge Regressor...")

    # Initialize model
    regressor = RidgeRegressor(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)

    # Fit model
    print("  Fitting model...")
    regressor.fit(train_img, train_meta, train_y)

    # Predict on validation set
    print("  Predicting on validation set...")
    val_preds = regressor.predict(val_img, val_meta)

    # Display sample predictions
    print(f"  Sample Predictions: {val_preds[:5]}")
    print(f"  Sample Actuals:     {val_y[:5]}")

    # Assertions
    # Predictions must be clipped to [1, 100]
    assert (val_preds >= 1.0).all() and (
        val_preds <= 100.0
    ).all(), "Predictions contain values outside the valid range [1, 100]"

    # Calculate RMSE manually to verify utils.compute_rmse
    rmse = compute_rmse(val_y, val_preds)
    print(f"  Validation RMSE: {rmse:.4f}")
    assert rmse > 0, "RMSE should be positive"

    print("  Regressor verification passed.")

    # ---------------------------------------------------------
    # 5. Full Pipeline Execution & Submission
    # ---------------------------------------------------------
    print("\n[5] Running Full Pipeline Wrapper (train_and_evaluate)...")
    # This function orchestrates training, validation, and saving the submission
    train_and_evaluate(data)

    # ---------------------------------------------------------
    # 6. Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission File Loaded: {sub_df.shape}")
    print(sub_df.head())

    # Assertions
    expected_rows = len(test_ids)  # Should match the test set size (subset in debug)
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    assert list(sub_df.columns) == [
        "Id",
        "Pawpularity",
    ], f"Invalid columns. Expected ['Id', 'Pawpularity'], got {list(sub_df.columns)}"

    assert not sub_df.isnull().values.any(), "Submission file contains NaN values"

    print("  Submission verification passed.")
    print("\n==================================================")
    print("       Demonstration Completed Successfully       ")
    print("==================================================")


if __name__ == "__main__":
    main()
