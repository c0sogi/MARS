import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import the library modules
sys.path.append(".")

from library.config import Config
from library import utils, dataset, model, engine, inference


def main():
    print("============================================================")
    print("      Cactus Identification: Library Demo & Verification    ")
    print("============================================================")

    # 1. Setup and Configuration Override
    # We override Config parameters to ensure the demo runs quickly (Speed Requirement)
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Set a fixed seed for the demo
    DEMO_SEED = 42
    utils.seed_everything(DEMO_SEED)

    # Override Config attributes
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 32  # Small batch size
    Config.DEBUG_SAMPLE_SIZE = 128  # Use only 128 samples for train/val/test
    Config.SEEDS = [DEMO_SEED]  # Ensemble only this single seed
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo stability

    # Re-run setup to ensure directories exist based on Config
    Config.setup()

    print(
        f"Configuration updated: Epochs={Config.NUM_EPOCHS}, "
        f"Batch Size={Config.BATCH_SIZE}, "
        f"Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading Pipeline...")

    # Get dataloaders with the debug sample size
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Fetch a single batch from the training loader
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Labels Shape: {labels.shape}")

    # Assertions to verify data integrity
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Expected image shape ({Config.BATCH_SIZE}, 3, 32, 32), got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected label shape ({Config.BATCH_SIZE}, 1), got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32 tensors"
    assert labels.dtype == torch.float32, "Labels should be float32 tensors"

    print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    net = model.CustomWideResNet()
    net = net.to(device)

    # Perform a forward pass with the batch fetched earlier
    images = images.to(device)
    with torch.no_grad():
        outputs = net(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify output shape (Batch Size, Num Classes) -> (32, 1)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {outputs.shape}"

    print("Model architecture verification passed.")

    # 4. Training Loop Execution
    print(f"\n[Step 4] Running Training Loop for Seed {DEMO_SEED}...")

    # Run the training engine
    # This handles the loop, validation, early stopping, and saving
    engine.run_training_seed(seed=DEMO_SEED, debug_sample_size=Config.DEBUG_SAMPLE_SIZE)

    # Verify that the model checkpoint was saved
    model_path = Config.get_model_path(DEMO_SEED)
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"

    print(f"Training completed. Model saved to: {model_path}")

    # 5. Inference and Submission
    print("\n[Step 5] Running Inference and Generating Submission...")

    # Run the inference engine
    # This loads the model(s) defined in Config.SEEDS and generates a CSV
    inference.ensemble_predictions(debug_sample_size=Config.DEBUG_SAMPLE_SIZE)

    # Verify submission file existence
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print("First 3 rows:")
    print(sub_df.head(3))

    # Assertions for submission validity
    assert (
        sub_df.shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {sub_df.shape[0]}"
    assert list(sub_df.columns) == [
        "id",
        "has_cactus",
    ], f"Expected columns ['id', 'has_cactus'], got {list(sub_df.columns)}"
    assert sub_df["has_cactus"].min() >= 0.0, "Probabilities cannot be negative"
    assert sub_df["has_cactus"].max() <= 1.0, "Probabilities cannot be > 1.0"

    print("Inference and submission verification passed.")
    print("\n============================================================")
    print("           Demo Completed Successfully!                     ")
    print("============================================================")


if __name__ == "__main__":
    main()
