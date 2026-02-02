import os
import torch
import pandas as pd
import numpy as np
from library import utils, data_loader, model, trainer


def run_demo():
    # 1. Setup and Initialization
    print(">>> Step 1: Setting up environment and seeds...")
    utils.seed_everything(42)
    device = utils.get_device()
    print(f"    Device detected: {device}")

    # 2. Data Loading Verification
    print("\n>>> Step 2: Verifying Data Loading...")
    batch_size = 8
    # We load cached data if available to speed up, otherwise it processes from scratch
    train_loader, val_loader, test_loader, ids_test = data_loader.get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(f"    Image batch shape: {images.shape}")
    print(f"    Angle batch shape: {angles.shape}")
    print(f"    Label batch shape: {labels.shape}")

    # Assertions
    # Expected shape: (Batch, 3, 75, 75)
    assert images.shape == (
        batch_size,
        3,
        75,
        75,
    ), f"Incorrect image shape: {images.shape}"
    assert angles.shape == (batch_size,), f"Incorrect angle shape: {angles.shape}"
    assert labels.shape == (batch_size,), f"Incorrect label shape: {labels.shape}"
    assert len(ids_test) > 0, "Test IDs list is empty"
    print("    Data shapes verified successfully.")

    # 3. Model Architecture Verification
    print("\n>>> Step 3: Verifying Model Architecture...")
    net = model.DRPPN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = net(images, angles)

    print(f"    Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (batch_size, 1), f"Incorrect output shape: {outputs.shape}"
    print("    Model forward pass verified successfully.")

    # 4. Full Pipeline Execution (Debug Mode)
    print("\n>>> Step 4: Running Full Training Pipeline (Debug Mode)...")
    # run_task with debug=True limits execution to 5 batches per epoch
    # This ensures the code runs quickly while testing the training loop and inference logic
    trainer.run_task(epochs=2, batch_size=16, debug=True)

    # 5. Submission Output Verification
    print("\n>>> Step 5: Verifying Submission Output...")
    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission file loaded. Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Assertions
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Missing required columns in submission"
    assert df_sub["is_iceberg"].min() >= 0.0, "Probabilities < 0 detected"
    assert df_sub["is_iceberg"].max() <= 1.0, "Probabilities > 1 detected"
    assert not df_sub.isnull().values.any(), "NaN values found in submission"

    # In debug mode, the number of predictions is limited by the LimitedLoader logic
    # (5 batches * 16 batch_size = 80 samples max, or less if dataset is smaller)
    # We just verify that we have some rows.
    assert len(df_sub) > 0, "Submission file is empty"

    print("    Submission format verified successfully.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
