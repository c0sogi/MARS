import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
import library.config as config
from library.utils import set_seed
from library.data_loader import load_dataset
from library.feature_engine import FeatureProcessor
from library.model_rf import train_rf_model
from library.model_mlp import train_mlp_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Demo Script...")

    # 1. Setup
    set_seed(config.RANDOM_STATE)

    # 2. Data Loading
    print("Loading datasets...")
    # Force reload from CSVs to ensure we have fresh data to subsample
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # OPTIMIZATION: Subsample data for speed
    # We use a small subset (e.g., 50 samples) to demonstrate the pipeline quickly.
    subset_size = 50
    print(f"Subsampling datasets to {subset_size} rows for rapid demonstration...")

    df_train = df_train.head(subset_size).copy()
    df_val = df_val.head(subset_size).copy()
    df_test = df_test.head(subset_size).copy()

    # Verify subsampling
    assert len(df_train) == subset_size
    assert len(df_val) == subset_size
    assert len(df_test) == subset_size

    # 3. Configuration Overrides for Speed
    print("Overriding hyperparameters for speed...")

    # RF Overrides
    config.RF_PARAMS["n_estimators"] = 10  # Reduce from 500 to 10
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead of multiprocessing for small data

    # MLP Overrides
    config.MLP_EPOCHS = 2  # Reduce from 50 to 2
    config.MLP_BATCH_SIZE = 8  # Smaller batch size for small data
    config.TFIDF_VOCAB_SIZE = 100  # Reduce vocab size for faster vectorization

    # 4. Feature Engineering
    print("Running Feature Engineering Pipeline...")
    processor = FeatureProcessor()

    # Note: load_cached_data=False ensures we process the subsampled data
    # instead of loading pre-computed features for the full dataset.
    rf_features, mlp_features, targets = processor.process(
        df_train, df_val, df_test, load_cached_data=False
    )

    # Verification of Feature Shapes
    print("Verifying feature shapes...")
    # RF features are concatenated arrays
    assert rf_features["train"].shape[0] == subset_size
    assert rf_features["test"].shape[0] == subset_size

    # MLP features are dictionaries
    assert mlp_features["train"]["semantic"].shape[0] == subset_size
    assert mlp_features["train"]["reliability"].shape[0] == subset_size
    assert mlp_features["train"]["community"].shape[0] == subset_size

    # Targets
    assert len(targets["train"]) == subset_size
    assert len(targets["val"]) == subset_size

    # 5. Train Random Forest (Stream A)
    print("Training Random Forest...")
    rf_val_preds, rf_test_preds, rf_model = train_rf_model(rf_features, targets)

    # Verify RF Predictions
    assert len(rf_val_preds) == subset_size
    assert len(rf_test_preds) == subset_size
    assert np.all(
        (rf_val_preds >= 0) & (rf_val_preds <= 1)
    ), "RF predictions out of bounds"

    # 6. Train MLP (Stream B)
    print("Training Topology-Aware MLP...")
    mlp_val_preds, mlp_test_preds, mlp_model = train_mlp_model(mlp_features, targets)

    # Verify MLP Predictions
    assert len(mlp_val_preds) == subset_size
    assert len(mlp_test_preds) == subset_size
    assert np.all(
        (mlp_val_preds >= 0) & (mlp_val_preds <= 1)
    ), "MLP predictions out of bounds"

    # 7. Ensemble
    print("Ensembling predictions...")
    # Simple average as per config weights (0.5/0.5)
    final_test_preds = 0.5 * rf_test_preds + 0.5 * mlp_test_preds

    assert len(final_test_preds) == subset_size

    # 8. Create Submission
    print("Generating submission file...")
    submission_df = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": final_test_preds,
        }
    )

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")

    # Final check
    assert os.path.exists(submission_path)
    loaded_sub = pd.read_csv(submission_path)
    assert len(loaded_sub) == subset_size
    assert list(loaded_sub.columns) == ["request_id", "requester_received_pizza"]

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
