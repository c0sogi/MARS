import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import get_data, get_fold_loaders, get_test_loader
from library.model import ACICNN
from library.train import train_one_epoch, validate
from library.train import train_model as run_full_pipeline


def demo_iceberg_classification():
    print("=== Starting Iceberg Classification Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override
    # ---------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    # Patch the Config class to run a lightweight version of the task
    Config.EPOCHS = 2  # Run only 2 epochs per fold for speed
    Config.NUM_FOLDS = 2  # Run only 2 folds (out of 5) to save time
    Config.BATCH_SIZE = 16  # Small batch size for demonstration
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Work Dir: {Config.WORK_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading and Processing Data...")

    # This will read JSONs from ./input and save processed .npy files to Config.WORK_DIR
    # Since WORK_DIR is new, this forces processing from scratch.
    data = get_data(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    angles_train = data["angles_train"]
    ids_test = data["ids_test"]

    print(f"    Train Images Shape: {X_train.shape}")
    print(f"    Train Labels Shape: {y_train.shape}")
    print(f"    Train Angles Shape: {angles_train.shape}")
    print(f"    Test IDs Count: {len(ids_test)}")

    # Validate Data Integrity
    assert (
        len(X_train) == len(y_train) == len(angles_train)
    ), "Mismatch in training data dimensions"
    assert (
        X_train.shape[1] == 3 and X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Incorrect image dimensions"
    assert not np.isnan(X_train).any(), "Input images contain NaNs"

    # ---------------------------------------------------------
    # 3. Fold Logic & Data Loaders
    # ---------------------------------------------------------
    print("\n[3] Verifying Fold Data Loaders (Fold 0)...")

    # Get loaders for the first fold. This handles leak-free scaling/imputation.
    train_loader, val_loader, scaler, imp_val = get_fold_loaders(
        fold_idx=0, data=data, batch_size=Config.BATCH_SIZE
    )

    print(f"    Scaler Mean: {scaler.mean_}")
    print(f"    Imputation Value (Angle): {imp_val}")

    # Fetch one batch to verify shapes
    batch_imgs, batch_raw_angs, batch_norm_angs, batch_labels = next(iter(train_loader))

    print(f"    Batch Image Shape: {batch_imgs.shape}")
    print(f"    Batch Raw Angle Shape: {batch_raw_angs.shape}")

    assert batch_imgs.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert batch_labels.shape == (Config.BATCH_SIZE,)
    assert batch_norm_angs.shape == (Config.BATCH_SIZE,)

    # ---------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Initializing Model and Testing Forward Pass...")

    model = ACICNN().to(device)

    # Move batch to device
    b_imgs = batch_imgs.to(device)
    b_raw = batch_raw_angs.to(device)
    b_norm = batch_norm_angs.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(b_imgs, b_raw, b_norm)

    print(f"    Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape is incorrect"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ---------------------------------------------------------
    # 5. Training & Validation Loop (Single Epoch Test)
    # ---------------------------------------------------------
    print("\n[5] Testing Training and Validation Logic (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train 1 epoch manually to verify logic
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.6f}")
    assert train_loss > 0, "Training loss should be positive"

    # Validate manually
    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.6f}")
    print(f"    Val Predictions Mean: {val_preds.mean():.4f}")

    assert len(val_preds) == len(val_targets)
    assert (val_preds >= 0.0).all() and (
        val_preds <= 1.0
    ).all(), "Predictions are not valid probabilities"

    # ---------------------------------------------------------
    # 6. Test Inference Setup
    # ---------------------------------------------------------
    print("\n[6] Testing Test Inference Loader...")

    # Get test loader using stats from Fold 0
    test_loader = get_test_loader(data, scaler, imp_val, batch_size=Config.BATCH_SIZE)

    test_batch_imgs, test_batch_raw, test_batch_norm = next(iter(test_loader))
    print(f"    Test Batch Image Shape: {test_batch_imgs.shape}")
    assert test_batch_imgs.shape[1:] == (3, 75, 75)

    # ---------------------------------------------------------
    # 7. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n[7] Executing Full Pipeline (2 Folds, 2 Epochs)...")
    print("    This utilizes the 'train_model' function from library.train")

    # Run the full pipeline which includes:
    # - Iterating folds (0 and 1 due to Config patch)
    # - Training
    # - Validation
    # - Checkpointing
    # - OOF Prediction
    # - Test Prediction Accumulation
    # - Submission Generation
    run_full_pipeline()

    # ---------------------------------------------------------
    # 8. Verification of Results
    # ---------------------------------------------------------
    print("\n[8] Verifying Submission Output...")

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    df_sub = pd.read_csv(sub_path)
    print(f"    Submission Rows: {len(df_sub)}")
    print(df_sub.head())

    assert len(df_sub) == len(ids_test), "Submission length mismatch"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_iceberg_classification()
