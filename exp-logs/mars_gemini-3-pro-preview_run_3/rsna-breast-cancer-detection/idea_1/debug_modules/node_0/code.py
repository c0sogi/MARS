import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import ResNet18Baseline
from library.trainer import Trainer
from library.inference import run_inference

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # --------------------------------------------------------------------------
    print("--- 1. Setup and Configuration ---")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config parameters for a fast demonstration
    # We modify class attributes directly to affect all modules using Config
    print("Overriding Config parameters for speed...")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.IMG_SIZE = (128, 128)  # Smaller image size for faster processing
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Size: {Config.IMG_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n--- 2. Verifying Data Pipeline ---")

    # Initialize DataLoaders
    dataloaders = get_dataloaders()

    # Check if all loaders are present
    assert "train" in dataloaders, "Train loader missing"
    assert "val" in dataloaders, "Val loader missing"
    assert "test" in dataloaders, "Test loader missing"

    train_loader = dataloaders["train"]

    # Fetch one batch to verify shapes
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch fetched successfully.")
        print(f"Image Batch Shape: {images.shape}")
        print(f"Label Batch Shape: {labels.shape}")

        # Verify shapes
        # Expected: (Batch, 3, H, W) because ResNet expects 3 channels
        expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        assert (
            images.shape == expected_shape
        ), f"Expected {expected_shape}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected ({Config.BATCH_SIZE},), got {labels.shape}"

        # Verify value range (should be normalized, but roughly within reasonable bounds)
        # Note: ImageNet normalization can produce negative values, so we just check type
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.float32, "Labels should be float32"

    except Exception as e:
        print(f"Data loading failed: {e}")
        raise e

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n--- 3. Verifying Model Architecture ---")

    # Instantiate model
    # We use pretrained=False here just to speed up initialization for the demo,
    # though the Trainer uses True by default.
    model = ResNet18Baseline(pretrained=False).to(Config.DEVICE)
    model.eval()

    # Move images to device
    images = images.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify output shape and values
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1)"
    assert (
        outputs.min() >= 0.0 and outputs.max() <= 1.0
    ), "Outputs must be probabilities in [0, 1]"

    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n--- 4. Demonstrating Training Loop ---")

    # Initialize Trainer
    trainer = Trainer()

    # Run fit for 1 epoch, limited to 2 batches per epoch for speed
    print("Starting short training run...")
    trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=Config.NUM_EPOCHS,
        max_batches=2,
    )

    # Verify checkpoint creation
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"Checkpoint successfully saved at: {Config.MODEL_CHECKPOINT_PATH}")
    else:
        # It's possible validation didn't improve if initialized randomly and run for 2 batches,
        # but with pretrained weights (default in Trainer) and proper loss, it usually saves once.
        # If it fails, we raise an error to investigate, though in a real demo we might be lenient.
        # However, the first validation loss is usually < infinity, so it should save.
        raise FileNotFoundError(
            f"Model checkpoint was not created at {Config.MODEL_CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # --------------------------------------------------------------------------
    print("\n--- 5. Demonstrating Inference Pipeline ---")

    # Run inference using the library function
    # Limits processing to 2 batches
    run_inference(output_path=Config.SUBMISSION_PATH, max_batches=2)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file generated at: {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("First few rows:")
        print(df_sub.head())

        # Verify columns
        required_cols = {"prediction_id", "cancer"}
        assert required_cols.issubset(
            df_sub.columns
        ), f"Missing columns. Found: {df_sub.columns}"

        # Verify values
        assert (
            df_sub["cancer"].min() >= 0.0 and df_sub["cancer"].max() <= 1.0
        ), "Predictions out of range [0, 1]"

    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n--- Demonstration Completed Successfully ---")
