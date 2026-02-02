import sys
import os
import pandas as pd
import numpy as np

from library.utils import set_seed, save_submission
from library.data_manager import DataManager
from library.model_wrapper import XGBClassifierWrapper


def main():
    # --- 1. Initialization ---
    print("Initializing demonstration...")
    set_seed(42)

    # --- 2. Data Management ---
    # Initialize DataManager with a specific cache directory for this run
    dm = DataManager(data_dir="./metadata", cache_dir="./working/demo_cache")

    print("Loading datasets (subsampled for speed)...")
    # Load a subset of 50,000 samples to ensure the script runs quickly
    dm.load_dataset(load_cached_data=False, sample_size=50000)

    # Validate data loading
    assert dm.train_df is not None, "Training data failed to load."
    assert dm.val_df is not None, "Validation data failed to load."
    assert dm.test_df is not None, "Test data failed to load."
    # Check that subsampling worked (allow for slightly less if dataset is smaller, though unlikely here)
    assert len(dm.train_df) <= 50000, "Training set subsampling failed."

    print(f"Train shape: {dm.train_df.shape}")
    print(f"Val shape:   {dm.val_df.shape}")
    print(f"Test shape:  {dm.test_df.shape}")

    # Encode targets
    print("Encoding target variable...")
    dm.encode_target()

    # Verify encoding
    unique_targets = dm.train_df["Cover_Type"].nunique()
    print(f"Number of unique classes: {unique_targets}")
    assert unique_targets > 1, "Target variable must have more than 1 class."

    # Create DMatrices
    print("Creating DMatrices...")
    dtrain = dm.get_dmatrix("train")
    dval = dm.get_dmatrix("val")
    dtest = dm.get_dmatrix("test")

    assert dtrain.num_row() == len(dm.train_df), "DMatrix row count mismatch (Train)."
    assert dtest.num_row() == len(dm.test_df), "DMatrix row count mismatch (Test)."

    # --- 3. Model Training ---
    print("Configuring XGBoost model...")
    # Override default parameters for a fast demonstration
    params = {
        "num_class": unique_targets,
        "eta": 0.2,  # Higher learning rate for faster convergence in demo
        "max_depth": 6,  # Moderate depth
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cuda",  # Use GPU
        "tree_method": "hist",
        "verbosity": 0,
    }

    model_wrapper = XGBClassifierWrapper(
        params=params,
        num_boost_round=100,  # Limit rounds for speed
        early_stopping_rounds=10,  # Stop early if no improvement
        verbose_eval=20,
    )

    print("Training model...")
    model_wrapper.train(dtrain, dval)

    # Validate training
    assert model_wrapper.model is not None, "Model failed to train."
    print("Training complete.")

    # --- 4. Prediction & Submission ---
    print("Generating predictions on test set...")
    preds_encoded = model_wrapper.predict(dtest)

    # Validate predictions shape
    assert len(preds_encoded) == len(dm.test_df), "Prediction length mismatch."

    # Inverse transform to get original labels
    preds_original = dm.inverse_transform_target(preds_encoded)

    # Get Test IDs
    test_ids = dm.get_test_ids()

    # Save submission
    output_path = "./submission/submission.csv"
    print(f"Saving submission to {output_path}...")
    save_submission(test_ids, preds_original, output_path)

    # --- 5. Final Validation ---
    print("Validating submission file...")
    assert os.path.exists(output_path), "Submission file was not created."

    df_sub = pd.read_csv(output_path)
    assert df_sub.shape == (
        len(dm.test_df),
        2,
    ), f"Submission shape mismatch. Expected {(len(dm.test_df), 2)}, got {df_sub.shape}"
    assert list(df_sub.columns) == ["Id", "Cover_Type"], "Submission columns mismatch."
    assert not df_sub.isnull().values.any(), "Submission contains null values."

    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
