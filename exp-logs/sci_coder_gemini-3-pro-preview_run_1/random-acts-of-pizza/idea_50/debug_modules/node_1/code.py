import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_splits
from library.feature_extractor import FeatureEngineer
from library.dataset import PizzaDataset
from library.models import RandomForestModel, FiLMClassifier
from library.engine import train_mlp, train_rf, predict_ensemble


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print("--- Setting up environment and configuration ---")
    seed_everything(Config.SEED)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for quick demonstration
    Config.MLP_EPOCHS = 2
    Config.RF_PARAMS["n_estimators"] = 10

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Data (Debug Mode)
    print("\n--- Loading Data ---")
    # This will load the CSVs and slice them to DEBUG_SAMPLE_SIZE
    train_df, val_df, test_df = get_splits(load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Verify data loading
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train data size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val data size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test data size mismatch"

    # 3. Feature Engineering
    print("\n--- Generating Features ---")
    fe = FeatureEngineer()

    # Force load_cached_data=False to ensure features match the DEBUG subset dimensions.
    # If we used True, it might load cached features from a full run, causing shape mismatches.
    (rf_train, rf_val, rf_test), (mlp_train, mlp_val, mlp_test) = fe.fit_transform(
        train_df, val_df, test_df, load_cached_data=False
    )

    print(f"RF Train Feature Shape: {rf_train.shape}")
    print(f"MLP Train Control Feature Shape: {mlp_train['control_features'].shape}")

    # Verify Feature Shapes
    assert rf_train.shape[0] == len(train_df), "RF train rows mismatch"
    assert mlp_train["title_emb"].shape[0] == len(train_df), "MLP train rows mismatch"

    # 4. Prepare Datasets and Loaders for MLP
    print("\n--- Preparing DataLoaders ---")
    target_col = "requester_received_pizza"

    # Extract labels
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    # Test set has no labels for prediction

    # Create Datasets
    train_dataset = PizzaDataset(mlp_train, y_train)
    val_dataset = PizzaDataset(mlp_val, y_val)
    test_dataset = PizzaDataset(mlp_test, labels=None)  # No labels for test

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 5. Initialize Models
    print("\n--- Initializing Models ---")

    # Random Forest
    rf_model = RandomForestModel()

    # FiLM MLP
    # Calculate control dimension from the generated features
    control_dim = mlp_train["control_features"].shape[1]
    mlp_model = FiLMClassifier(control_input_dim=control_dim).to(device)

    print(f"MLP Control Input Dimension: {control_dim}")

    # 6. Train Models
    print("\n--- Training Random Forest ---")
    train_rf(rf_model, rf_train, y_train)

    print("\n--- Training MLP ---")
    mlp_model, best_auc = train_mlp(mlp_model, train_loader, val_loader, device)
    print(f"Best MLP Validation AUC: {best_auc:.4f}")

    # 7. Generate Predictions and Submission
    print("\n--- Generating Submission ---")
    predict_ensemble(
        rf_model=rf_model,
        mlp_model=mlp_model,
        rf_test_feats=rf_test,
        mlp_test_loader=test_loader,
        test_df=test_df,
        device=device,
    )

    # 8. Final Verification
    print("\n--- Verifying Submission ---")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    submission = pd.read_csv(submission_path)

    # Check shape
    expected_rows = len(test_df)
    if len(submission) != expected_rows:
        raise AssertionError(
            f"Submission has {len(submission)} rows, expected {expected_rows}"
        )

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    if list(submission.columns) != expected_cols:
        raise AssertionError(f"Submission columns mismatch. Got {submission.columns}")

    # Check value range
    probs = submission["requester_received_pizza"]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Probabilities out of range [0, 1]")

    # Check IDs match
    if not submission["request_id"].equals(test_df["request_id"]):
        raise AssertionError("Request IDs in submission do not match test data")

    print("Verification successful! Demo completed.")


if __name__ == "__main__":
    run_demo()
