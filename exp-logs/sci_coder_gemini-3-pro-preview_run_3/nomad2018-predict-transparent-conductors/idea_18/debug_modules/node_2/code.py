import os
import pandas as pd
import numpy as np
import shutil
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    INPUT_DIR,
    TARGET_COLS,
    RANDOM_SEED,
)
from library.features import process_dataset
from library.model import DualTargetRegressor


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup paths for small demo data
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    demo_train_meta_path = os.path.join(WORKING_DIR, "demo_train_metadata.csv")
    demo_val_meta_path = os.path.join(WORKING_DIR, "demo_val_metadata.csv")

    demo_train_features_path = os.path.join(WORKING_DIR, "demo_train_features.parquet")
    demo_val_features_path = os.path.join(WORKING_DIR, "demo_val_features.parquet")

    # 2. Create small subsets of metadata for speed
    print(f"Creating subsampled metadata in {WORKING_DIR}...")

    # Load full metadata
    df_train_full = pd.read_csv(train_meta_path)
    df_val_full = pd.read_csv(val_meta_path)

    # Sample 50 rows for training, 10 for validation
    df_train_small = df_train_full.head(50).copy()
    df_val_small = df_val_full.head(10).copy()

    # Save small metadata
    df_train_small.to_csv(demo_train_meta_path, index=False)
    df_val_small.to_csv(demo_val_meta_path, index=False)

    print(f"Train subset shape: {df_train_small.shape}")
    print(f"Val subset shape: {df_val_small.shape}")

    # 3. Feature Engineering (using library.features.process_dataset)
    # This function reads the metadata, loads XYZ files from INPUT_DIR, computes features, and saves parquet.
    print("\n--- Generating Features ---")

    # Force re-computation by setting load_cached_data=False (or ensure paths don't exist)
    if os.path.exists(demo_train_features_path):
        os.remove(demo_train_features_path)
    if os.path.exists(demo_val_features_path):
        os.remove(demo_val_features_path)

    print("Processing training subset...")
    df_train_features = process_dataset(
        metadata_path=demo_train_meta_path,
        output_path=demo_train_features_path,
        load_cached_data=False,
    )

    print("Processing validation subset...")
    df_val_features = process_dataset(
        metadata_path=demo_val_meta_path,
        output_path=demo_val_features_path,
        load_cached_data=False,
    )

    # Verify feature generation
    assert os.path.exists(
        demo_train_features_path
    ), "Train features parquet not created."
    assert not df_train_features.empty, "Train features dataframe is empty."
    # Check for some expected columns (e.g., global_volume from features.py)
    assert (
        "global_volume" in df_train_features.columns
    ), "Feature 'global_volume' missing."
    print("Feature generation successful.")

    # 4. Model Training (using library.model.DualTargetRegressor)
    print("\n--- Training Model ---")

    # Define fast hyperparameters for demo
    fast_params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": 1,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
    }

    model = DualTargetRegressor(params=fast_params)

    # fit() handles data preparation (dropping non-feature cols) and log-transform of targets internally
    model.fit(df_train_features, df_val_features, verbose=True)

    print("Model training complete.")

    # 5. Inference and Validation
    print("\n--- Inference ---")

    # Predict on validation set
    preds = model.predict(df_val_features)

    print("Predictions head:")
    print(preds.head())

    # Verify predictions
    assert isinstance(preds, pd.DataFrame), "Predictions should be a DataFrame."
    assert len(preds) == len(
        df_val_features
    ), f"Expected {len(df_val_features)} predictions, got {len(preds)}."
    assert (
        list(preds.columns) == TARGET_COLS
    ), f"Columns mismatch. Expected {TARGET_COLS}, got {list(preds.columns)}."

    # Check for non-negative values (physical constraint handled by inverse_transform_targets)
    if (preds < 0).any().any():
        raise AssertionError(
            "Predictions contain negative values, which violates physical constraints."
        )

    # Check basic accuracy (sanity check - just ensuring it's not returning garbage/NaNs)
    assert not preds.isnull().any().any(), "Predictions contain NaNs."

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
