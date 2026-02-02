import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import get_model
from library.train import Trainer
from library.predict import generate_submission


def main():
    print("Starting Herbarium 2020 Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo Speed
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to run a fast debug session.
    print("\n[Step 1] Configuring for fast execution...")

    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"  DEBUG_SAMPLE_SIZE: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  EPOCHS: {Config.EPOCHS}")
    print(f"  WORKING_DIR: {Config.WORKING_DIR}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")

    # We pass load_cached_data=False to force re-computation of sampler weights
    # for our small subset, rather than loading cached weights for the full dataset.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify dataset lengths
    assert (
        len(train_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_loader.dataset)}"
    assert (
        len(val_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_loader.dataset)}"

    # Verify Batch Shape
    images, labels = next(iter(train_loader))
    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert labels.max() < Config.NUM_CLASSES, "Label index out of bounds"

    print("  Data Loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    model = get_model(pretrained=False)  # No need to download weights for shape check
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("  Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Run fit. This will train for 1 epoch on the subset.
    # We expect it to save a checkpoint.
    trainer.fit(load_cached_data=False)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Checkpoint file not found at {Config.MODEL_CHECKPOINT_PATH}"

    print("  Training loop completed and checkpoint saved.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    # Use the predict module's function
    # We pass the test_loader we already created to save time
    generate_submission(test_loader=test_loader)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Head:\n{df_sub.head()}")

    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission file missing required columns."
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    print("  Submission generation verified successfully.")

    print("\nAll library components demonstrated and verified successfully.")


if __name__ == "__main__":
    main()
