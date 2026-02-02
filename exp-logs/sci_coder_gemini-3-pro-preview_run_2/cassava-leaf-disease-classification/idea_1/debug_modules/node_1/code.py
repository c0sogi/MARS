import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import get_model
from library.engine import train_model, predict


def run_demo():
    print("Initializing demonstration...")

    # 1. Define Custom Configuration for Fast Execution
    class FastConfig(Config):
        # Enable debug mode to use a small subset of data (e.g., 50 samples)
        DEBUG = True
        DEBUG_SAMPLE_SIZE = 50

        # Reduce training parameters for speed
        NUM_EPOCHS = 1
        BATCH_SIZE = 8

        # Use a separate directory for this demo
        WORKING_DIR = "./working/demo_run"
        SUBMISSION_DIR = "./working/demo_submission"

        # Update paths based on new directories
        MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "demo_model.pth")
        SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

        def __init__(self):
            # Call parent init to create directories
            super().__init__()

    # Instantiate config
    cfg = FastConfig()

    # Set seed for reproducibility
    seed_everything(cfg.SEED)
    print("Configuration and seeding complete.")

    # 2. Verify Data Loading
    print("\n--- Verifying Data Pipeline ---")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        cfg.BATCH_SIZE,
        3,
        cfg.IMG_SIZE,
        cfg.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (cfg.BATCH_SIZE,), "Incorrect label tensor shape"
    print("Data pipeline verification passed.")

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    device = torch.device(cfg.DEVICE)
    model = get_model(cfg.NUM_CLASSES, device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        cfg.BATCH_SIZE,
        cfg.NUM_CLASSES,
    ), "Incorrect model output shape"
    print("Model architecture verification passed.")

    # 4. Run Training Pipeline
    print("\n--- Running Training Loop (Fast Mode) ---")
    # This will train for 1 epoch on 50 samples
    trained_model = train_model(cfg)

    # Verify model file creation
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        print(f"Model successfully saved to: {cfg.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model file was not saved after training.")

    # 5. Run Inference Pipeline
    print("\n--- Running Inference Pipeline ---")
    submission_df = predict(cfg)

    # Verify submission file
    if os.path.exists(cfg.SUBMISSION_PATH):
        print(f"Submission file successfully saved to: {cfg.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not saved after inference.")

    # Verify submission content
    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission columns: {submission_df.columns.tolist()}")

    # Check if number of rows matches debug sample size
    # Note: get_dataloaders truncates test set to DEBUG_SAMPLE_SIZE in debug mode
    assert (
        len(submission_df) == cfg.DEBUG_SAMPLE_SIZE
    ), f"Expected {cfg.DEBUG_SAMPLE_SIZE} predictions, got {len(submission_df)}"

    assert list(submission_df.columns) == [
        "image_id",
        "label",
    ], "Incorrect submission columns"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
