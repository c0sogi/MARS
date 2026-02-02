import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission, load_metadata_splits
from library.rf_manager import RFManager
from library.mlp_manager import MLPManager


def create_mini_dataset():
    """
    Creates a small subset of the metadata to ensure the demo runs quickly.
    """
    print("Creating mini dataset for rapid demonstration...")

    # Define mini metadata directory
    mini_dir = os.path.join(Config.WORKING_DIR, "demo_metadata")
    os.makedirs(mini_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample subsets (e.g., 20 samples each)
    # We ensure we have both classes in train/val for valid AUC calculation
    n_samples = 20

    # Stratified sample for train/val if possible, else just head
    mini_train = (
        train_df.groupby(Config.TARGET_COL, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_samples // 2), random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    mini_val = (
        val_df.groupby(Config.TARGET_COL, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_samples // 2), random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    mini_test = test_df.head(n_samples)

    # Save mini metadata
    mini_train_path = os.path.join(mini_dir, "train.csv")
    mini_val_path = os.path.join(mini_dir, "val.csv")
    mini_test_path = os.path.join(mini_dir, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path, len(mini_test)


def override_config(train_path, val_path, test_path):
    """
    Overrides Config parameters for speed and to point to the mini dataset.
    """
    print("Overriding configuration for speed...")

    # Paths
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Artifacts
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, "demo_artifacts")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Random Forest Parameters
    Config.RF_N_ESTIMATORS = 10  # Reduced from 500
    Config.RF_N_JOBS = 1  # Avoid overhead for small data

    # MLP Parameters
    Config.MLP_EPOCHS = 2  # Reduced from 50
    Config.MLP_BATCH_SIZE = 4  # Small batch for small data
    Config.MLP_HIDDEN_DIM = 32  # Smaller network
    Config.MLP_PATIENCE = 1  # Early stopping check

    # Feature Engineering
    Config.TFIDF_MAX_FEATURES = 100  # Reduced vocab
    Config.TOP_K_SUBREDDITS = 5  # Reduced top-k


def run_demo():
    # 1. Setup
    set_seed(42)

    # 2. Prepare Data & Config
    train_path, val_path, test_path, test_size = create_mini_dataset()
    override_config(train_path, val_path, test_path)

    # Verify Data Loading
    train_df, val_df, test_df = load_metadata_splits()
    print(
        f"Loaded Mini Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    assert len(train_df) > 0, "Training set is empty"
    assert len(test_df) == test_size, "Test set size mismatch"

    # 3. Stream A: Random Forest Pipeline
    print("\n" + "=" * 30)
    print("Running Stream A: Random Forest")
    print("=" * 30)

    rf_manager = RFManager()

    # Train
    # We set load_cached_data=False to force execution of the feature engineering pipeline
    rf_auc = rf_manager.train(load_cached_data=False)
    print(f"RF Demo AUC: {rf_auc:.4f}")

    # Validate Logic
    assert isinstance(rf_auc, float), "RF AUC should be a float"
    assert 0.0 <= rf_auc <= 1.0, "RF AUC out of bounds"

    # Predict
    rf_preds = rf_manager.predict_test(
        load_cached_data=True
    )  # Use cache generated during train
    print(f"RF Predictions Shape: {rf_preds.shape}")
    assert len(rf_preds) == test_size, "RF Predictions length mismatch"
    assert np.all((rf_preds >= 0) & (rf_preds <= 1)), "RF probabilities out of bounds"

    # 4. Stream B: MLP Pipeline
    print("\n" + "=" * 30)
    print("Running Stream B: Neural Network")
    print("=" * 30)

    mlp_manager = MLPManager()

    # Train
    # We can use cached data here because RFManager already triggered the processing
    # for the same split names ('train', 'val', 'test') in the same Config.IDEA_DIR.
    mlp_auc = mlp_manager.train(load_cached_data=True)
    print(f"MLP Demo AUC: {mlp_auc:.4f}")

    # Validate Logic
    assert isinstance(mlp_auc, float), "MLP AUC should be a float"
    # Note: On very small random data, AUC might be 0.0 or 1.0, checking bounds is sufficient
    assert 0.0 <= mlp_auc <= 1.0, "MLP AUC out of bounds"

    # Predict
    mlp_preds = mlp_manager.predict_test(load_cached_data=True)
    print(f"MLP Predictions Shape: {mlp_preds.shape}")
    assert len(mlp_preds) == test_size, "MLP Predictions length mismatch"

    # 5. Ensemble & Submission
    print("\n" + "=" * 30)
    print("Ensembling and Submitting")
    print("=" * 30)

    # Simple Weighted Average
    final_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_preds) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_preds
    )

    # Generate Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    request_ids = test_df[Config.ID_COL].values

    save_submission(request_ids, final_preds, filename=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"
    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (test_size, 2), f"Submission shape mismatch: {sub_df.shape}"
    assert Config.ID_COL in sub_df.columns, "ID column missing in submission"
    assert Config.TARGET_COL in sub_df.columns, "Target column missing in submission"

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
