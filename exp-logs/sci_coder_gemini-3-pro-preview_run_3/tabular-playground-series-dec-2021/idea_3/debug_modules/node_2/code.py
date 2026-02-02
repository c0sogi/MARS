import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_accuracy
from library.data_processing import get_dataloaders, CoverTypeDataset
from library.model import DCNV2
from library.train import Trainer


def main():
    print("=== Starting Library Code Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override Config parameters to ensure fast execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use a specific working directory for this demo to avoid messing with existing caches if needed
    # However, we will use the default logic but force re-processing for the demo subset
    debug_size = 1000

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: EPOCHS=1, BATCH_SIZE=128, DEBUG_SIZE=1000")

    # --------------------------------------------------------------------------
    # 2. Data Processing Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Pipeline (get_dataloaders)...")

    # We set load_cached_data=False to demonstrate the feature engineering logic works
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False, debug_sample_size=debug_size
    )

    # Fetch a single batch to inspect
    x_cont, x_cat, target = next(iter(train_loader))

    print(
        f"Batch Shapes -> Continuous: {x_cont.shape}, Categorical: {x_cat.shape}, Target: {target.shape}"
    )

    # Assertions
    assert (
        len(train_loader.dataset) == debug_size
    ), f"Train dataset size mismatch. Expected {debug_size}, got {len(train_loader.dataset)}"
    assert x_cont.dim() == 2, "Continuous features should be 2D (Batch, Features)"
    assert x_cat.dim() == 2, "Categorical features should be 2D (Batch, 2)"
    assert x_cat.shape[1] == 2, "Expected 2 categorical columns (Soil, Wilderness)"
    assert target.dim() == 1, "Target should be 1D"

    # Determine number of continuous features for model init
    num_cont_features = x_cont.shape[1]
    print(f"Detected {num_cont_features} continuous features.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture (DCNV2)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNV2(num_cont_features=num_cont_features).to(device)

    # Move batch to device
    x_cont_dev = x_cont.to(device)
    x_cat_dev = x_cat.to(device)

    # Forward pass
    logits = model(x_cont_dev, x_cat_dev)

    print(f"Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        x_cont.shape[0],
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected ({x_cont.shape[0]}, {Config.NUM_CLASSES}), got {logits.shape}"

    # Verify calculate_accuracy utility
    acc = calculate_accuracy(logits, target.to(device))
    print(f"Initial (Untrained) Batch Accuracy: {acc:.4f}")
    assert 0.0 <= acc <= 1.0, "Accuracy must be between 0 and 1"

    # --------------------------------------------------------------------------
    # 4. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Trainer (Fit & Predict)...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_ids=test_ids,
        device=device,
    )

    # Run training (1 epoch as configured)
    print("Running trainer.fit()...")
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # Check if model file was saved (Training saves 'dcn_model.pth' on improvement)
    # Since we only run 1 epoch, it might save if val_acc > 0.
    # We won't strictly assert file existence as 1 epoch might not trigger save if acc is 0,
    # but with 7 classes random chance is > 0.

    # Run prediction
    print("Running trainer.predict()...")
    trainer.predict()

    # --------------------------------------------------------------------------
    # 5. Submission Verification
    # --------------------------------------------------------------------------
    print("\n[Step 5] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Assertions
    assert (
        df_sub.shape[0] == debug_size
    ), f"Submission row count mismatch. Expected {debug_size}, got {df_sub.shape[0]}"
    assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column {Config.TARGET_COL}"

    # Verify values are within valid class range (1-7)
    # Note: Model predicts 0-6 internally, converted to 1-7 in predict()
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 1 and preds.max() <= 7
    ), "Predictions out of valid range [1, 7]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
