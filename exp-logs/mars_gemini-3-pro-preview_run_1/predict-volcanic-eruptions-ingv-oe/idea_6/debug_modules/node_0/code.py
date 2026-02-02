import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureProcessor
from library.model_tabular import run_lgbm_cv
from library.model_vision import run_vision_cv
from library.model_stacking import run_stacking

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(n_train=20, n_val=10, n_test=10):
    """
    Creates small subsets of the metadata to allow the pipeline
    to run end-to-end quickly for demonstration.
    """
    print(
        f"Creating mini metadata sets (Train={n_train}, Val={n_val}, Test={n_test})..."
    )

    # Read original metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample subsets
    df_train_mini = df_train_full.head(n_train).copy()
    df_val_mini = df_val_full.head(n_val).copy()
    df_test_mini = df_test_full.head(n_test).copy()

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Save to working directory
    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def main():
    # 1. Setup and Reproducibility
    print("=== Starting Pipeline Demonstration ===")
    seed_everything(Config.SEED)

    # 2. Patch Configuration for Speed
    # We modify the Config class attributes directly to affect all downstream modules
    print("Patching configuration for fast execution...")
    Config.NN_PARAMS["epochs"] = 1
    Config.NN_PARAMS["batch_size"] = 4
    Config.NN_PARAMS["num_workers"] = 0  # Avoid multiprocessing overhead in demo
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["verbose"] = -1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Prepare Data
    mini_train_path, mini_val_path, mini_test_path = create_mini_metadata()

    # Update Config paths to point to mini metadata for the stacking phase later
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # 4. Feature Engineering
    print("\n=== Step 1: Feature Engineering ===")
    processor = FeatureProcessor()

    # We process the mini-datasets but name them 'train', 'val', 'test'
    # so the model modules find the files at the expected locations (e.g., train_features.parquet)
    print("Processing Mini Train Set...")
    processor.process_data(mini_train_path, "train", load_cached_data=False)

    print("Processing Mini Val Set...")
    processor.process_data(mini_val_path, "val", load_cached_data=False)

    print("Processing Mini Test Set...")
    processor.process_data(mini_test_path, "test", load_cached_data=False)

    # Verify files were created
    expected_files = [
        "train_features.parquet",
        "train_spectrograms.npy",
        "train_targets.npy",
        "val_features.parquet",
        "val_spectrograms.npy",
        "val_targets.npy",
        "test_features.parquet",
        "test_spectrograms.npy",
    ]
    for fname in expected_files:
        fpath = os.path.join(Config.WORKING_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Feature engineering failed to create {fname}")

    print("Feature engineering verification passed.")

    # 5. Tabular Branch (LightGBM)
    print("\n=== Step 2: Tabular Model (LightGBM) ===")
    # Run CV in debug mode (though we already patched config, debug mode adds extra safety)
    df_oof_tab, df_test_tab = run_lgbm_cv(debug=True)

    # Verify outputs
    assert len(df_oof_tab) > 0, "Tabular OOF is empty"
    assert len(df_test_tab) > 0, "Tabular Test Preds is empty"
    assert "time_to_eruption" in df_oof_tab.columns
    print("Tabular branch execution successful.")

    # 6. Vision Branch (EfficientNet)
    print("\n=== Step 3: Vision Model (EfficientNet) ===")
    # Run CV in debug mode
    df_oof_vis, df_test_vis = run_vision_cv(debug=True)

    # Verify outputs
    assert len(df_oof_vis) > 0, "Vision OOF is empty"
    assert len(df_test_vis) > 0, "Vision Test Preds is empty"
    assert "time_to_eruption" in df_oof_vis.columns
    print("Vision branch execution successful.")

    # 7. Stacking (Meta-Learner)
    print("\n=== Step 4: Stacking & Submission ===")
    run_stacking(debug=True)

    # 8. Final Validation
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    print("\n=== Final Validation ===")
    print(f"Submission File: {submission_path}")
    print(f"Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check format
    expected_cols = ["segment_id", "time_to_eruption"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check if predictions are numeric and non-negative
    if not np.issubdtype(df_sub["time_to_eruption"].dtype, np.number):
        raise TypeError("Predictions are not numeric.")

    if (df_sub["time_to_eruption"] < 0).any():
        raise ValueError("Found negative predictions in submission.")

    print("\nPipeline demonstration completed successfully!")


if __name__ == "__main__":
    main()
