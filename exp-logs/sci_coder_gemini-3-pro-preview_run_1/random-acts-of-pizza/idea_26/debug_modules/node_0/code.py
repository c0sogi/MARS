import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# 1. Import Config and Patch for Speed/Demo purposes
# We must patch Config BEFORE importing other modules because some default arguments
# in those modules are evaluated at import time.
from library.config import Config

# Define a demo working directory
DEMO_DIR = "./working/demo_execution"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Patch Configuration
print("Patching Configuration for Demo...")
Config.WORKING_DIR = DEMO_DIR
Config.CACHE_DIR = DEMO_DIR
Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Optimize hyperparameters for speed
Config.RF_ESTIMATORS = 10  # Reduced from 500
Config.MLP_EPOCHS = 2  # Reduced from 50
Config.MLP_BATCH_SIZE = 16  # Smaller batch size
Config.MLP_HIDDEN_DIM = 64  # Smaller model
Config.TOP_K_SUBREDDITS = 10  # Fewer features
Config.MLP_PATIENCE = 1  # Fail fast

# 2. Import remaining library components
from library.utils import set_seed, save_submission
from library.feature_engineering import FeatureEngineer
from library.model_rf import train_rf, predict_rf
from library.model_nn import train_nn_model, predict_nn_model

if __name__ == "__main__":
    # Set global seed
    set_seed(Config.SEED)

    # ==========================================
    # Step 1: Feature Engineering
    # ==========================================
    print("\n=== Step 1: Feature Engineering ===")
    fe = FeatureEngineer()

    # We set load_cached_data=False to demonstrate the generation logic,
    # but in a real run with existing cache, True is preferred.
    # Since we changed CACHE_DIR to a new empty folder, it will compute from scratch.
    data_dict = fe.run(load_cached_data=False)

    # Validate Data Structure
    assert "rf" in data_dict, "Missing RF data"
    assert "mlp" in data_dict, "Missing MLP data"
    assert "ids" in data_dict, "Missing ID data"

    # Unpack RF Data
    X_rf_train, y_rf_train, X_rf_val, y_rf_val, X_rf_test = data_dict["rf"]
    print(f"RF Train Data Shape: {X_rf_train.shape}")
    assert X_rf_train.shape[0] == len(y_rf_train), "RF Train mismatch"

    # Unpack MLP Data
    mlp_data = data_dict["mlp"]
    train_tuple = mlp_data["train"]  # (title, body, hist, meta, y)
    val_tuple = mlp_data["val"]
    test_tuple = mlp_data["test"]
    sub_emb = mlp_data["sub_emb"]

    print(f"MLP Subreddit Embeddings Shape: {sub_emb.shape}")
    assert len(train_tuple) == 5, "MLP Train tuple invalid"

    # ==========================================
    # Step 2: Random Forest Model
    # ==========================================
    print("\n=== Step 2: Training Random Forest ===")
    # Train
    rf_model = train_rf(
        X_rf_train, y_rf_train, X_rf_val, y_rf_val, n_estimators=Config.RF_ESTIMATORS
    )

    # Predict
    rf_preds_val = predict_rf(rf_model, X_rf_val)
    rf_preds_test = predict_rf(rf_model, X_rf_test)

    # Validate Predictions
    print(f"RF Test Predictions Mean: {rf_preds_test.mean():.4f}")
    assert (
        rf_preds_test.min() >= 0 and rf_preds_test.max() <= 1
    ), "RF preds out of range"
    assert len(rf_preds_test) == len(data_dict["ids"]), "RF pred length mismatch"

    # ==========================================
    # Step 3: Neural Network Model
    # ==========================================
    print("\n=== Step 3: Training Neural Network ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Train
    nn_model = train_nn_model(
        train_tuple,
        val_tuple,
        sub_emb,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        epochs=Config.MLP_EPOCHS,
        batch_size=Config.MLP_BATCH_SIZE,
        device=device,
    )

    # Predict
    nn_preds_test = predict_nn_model(
        nn_model, test_tuple, batch_size=Config.MLP_BATCH_SIZE, device=device
    )

    # Validate Predictions
    print(f"NN Test Predictions Mean: {nn_preds_test.mean():.4f}")
    assert (
        nn_preds_test.min() >= 0 and nn_preds_test.max() <= 1
    ), "NN preds out of range"
    assert len(nn_preds_test) == len(data_dict["ids"]), "NN pred length mismatch"

    # ==========================================
    # Step 4: Ensemble & Submission
    # ==========================================
    print("\n=== Step 4: Ensembling and Submission ===")

    # Simple Average Ensemble
    final_preds = 0.5 * rf_preds_test + 0.5 * nn_preds_test

    # Save Submission
    save_submission(data_dict["ids"], final_preds, Config.SUBMISSION_FILE)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"
    df_check = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_check)}")
    print(df_check.head())

    assert list(df_check.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Invalid columns"
    assert len(df_check) == 1162, "Expected 1162 test samples"  # Based on dataset info

    print("\nDemo Execution Completed Successfully!")
