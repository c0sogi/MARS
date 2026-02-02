import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, _load_and_process_data
from library.model import DSICNN
from library.train import train_one_epoch, validate, predict


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("--- 1. Initializing and Overriding Configuration ---")
    seed_everything(Config.SEED)

    # Override Config for rapid execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_FOLDS = 2

    # Ensure working directory exists (handled by Config import, but good to be safe)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    logger = get_logger("demo_script")
    logger.info(f"Device: {Config.DEVICE}")

    # 2. Data Pipeline Verification
    print("\n--- 2. Verifying Data Loading and Processing ---")

    # Test internal data loading logic
    # We force reload to ensure processing logic is exercised
    data_map = _load_and_process_data(load_cached_data=False)

    X_train = data_map["X_train"]
    y_train = data_map["y_train"]
    angles_train = data_map["meta_train"]

    # Assertions for data shapes
    # Image shape: (N, 3, 75, 75)
    assert len(X_train.shape) == 4, f"Expected 4D X_train, got {X_train.shape}"
    assert X_train.shape[1] == 3, f"Expected 3 channels, got {X_train.shape[1]}"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Expected 75x75 spatial dims"

    # Label shape: (N,)
    assert len(y_train.shape) == 1, f"Expected 1D y_train, got {y_train.shape}"
    assert len(X_train) == len(y_train), "Mismatch between X and y lengths"

    # Angle shape: (N,)
    assert len(angles_train.shape) == 1, "Expected 1D angles"
    assert len(angles_train) == len(X_train), "Mismatch between X and angles"

    logger.info("Raw data shapes verified successfully.")

    # Test DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch to verify Tensor conversion
    images, angles, labels = next(iter(train_loader))

    assert isinstance(images, torch.Tensor), "Loader should return image tensors"
    assert isinstance(angles, torch.Tensor), "Loader should return angle tensors"
    assert isinstance(labels, torch.Tensor), "Loader should return label tensors"

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Unexpected batch shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label batch shape: {labels.shape}"

    logger.info("DataLoader batch verification passed.")

    # 3. Model Architecture Verification
    print("\n--- 3. Verifying Model Architecture ---")

    model = DSICNN().to(Config.DEVICE)

    # Create dummy inputs
    dummy_img = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(Config.DEVICE)
    dummy_ang = torch.randn(Config.BATCH_SIZE).to(Config.DEVICE)

    # Forward pass
    output = model(dummy_img, dummy_ang)

    # Check output shape: (Batch, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output (B, 1), got {output.shape}"

    # Check if output is finite (no NaNs)
    assert torch.all(torch.isfinite(output)), "Model output contains NaNs or Infs"

    logger.info("Model forward pass verified successfully.")

    # 4. Training Loop Demonstration
    print("\n--- 4. Verifying Training Loop ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for one epoch
    logger.info("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )

    assert isinstance(train_loss, float), "Train loss must be a float"
    assert train_loss > 0, "Train loss should be positive"
    logger.info(f"Train Loss: {train_loss:.4f}")

    # Validate
    logger.info("Running validate...")
    val_loss = validate(model, val_loader, criterion, Config.DEVICE)

    assert isinstance(val_loss, float), "Val loss must be a float"
    assert val_loss > 0, "Val loss should be positive"
    logger.info(f"Val Loss: {val_loss:.4f}")

    # 5. Inference Demonstration
    print("\n--- 5. Verifying Inference ---")

    logger.info("Running predict on test set...")
    predictions = predict(model, test_loader, Config.DEVICE)

    # Check predictions shape matches test set size
    test_set_size = len(test_loader.dataset)
    assert predictions.shape == (
        test_set_size,
    ), f"Expected {test_set_size} predictions, got {predictions.shape}"

    # Check probability range [0, 1]
    assert np.all(predictions >= 0.0) and np.all(
        predictions <= 1.0
    ), "Predictions must be probabilities in [0, 1]"

    logger.info(f"Generated {len(predictions)} predictions successfully.")

    # 6. Submission File Generation (Mock)
    print("\n--- 6. Verifying Submission Generation ---")
    # We just create a dummy dataframe to show we can link IDs to preds
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    assert len(df_test_meta) == len(
        predictions
    ), "Metadata length mismatch with predictions"

    submission = pd.DataFrame({"id": df_test_meta["id"], "is_iceberg": predictions})

    # Verify structure
    assert "id" in submission.columns
    assert "is_iceberg" in submission.columns
    assert len(submission) == test_set_size

    print("Demo completed successfully. All components verified.")


if __name__ == "__main__":
    run_demo()
