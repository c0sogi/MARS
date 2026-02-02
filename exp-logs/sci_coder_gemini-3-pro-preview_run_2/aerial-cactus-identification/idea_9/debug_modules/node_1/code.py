import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import DeeplySupervisedUNet
from library.train import run_training
from library.inference import generate_submission


def main():
    print("Starting Demonstration Script...")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demonstration
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Reduce dataset size for quick execution (Debug Mode)
    Config.DEBUG_SUBSET_SIZE = 200  # Only use 200 images

    # Reduce training duration
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for the small subset
    Config.SEEDS = [42]  # Run only one seed instead of the full ensemble

    # Ensure reproducibility
    set_seed(42)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Epochs: {Config.MAX_EPOCHS}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Force reload from scratch (ignore existing cache) by setting load_cached_data=False
    # Note: In a real run, we'd keep True. Here we test the processing logic.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Channel count mismatch (should be 3 for RGB)"
    assert (
        images.shape[2] == Config.IMAGE_SIZE and images.shape[3] == Config.IMAGE_SIZE
    ), f"Image resolution mismatch. Expected {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}"
    assert labels.ndim == 1, "Labels should be a 1D tensor"

    # Verify Test Loader (returns images and ids)
    test_images, test_ids = next(iter(test_loader))
    print(f"Test Batch - Images: {test_images.shape}, IDs length: {len(test_ids)}")
    assert len(test_ids) == Config.BATCH_SIZE, "Test ID batch size mismatch"

    print("Data Loading verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeeplySupervisedUNet().to(device)

    # Pass a dummy batch
    dummy_input = images.to(device)
    semantic_logits, detail_logits = model(dummy_input)

    print(
        f"Model Output Shapes - Semantic: {semantic_logits.shape}, Detail: {detail_logits.shape}"
    )

    # Assertions
    assert semantic_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Semantic head output shape mismatch"
    assert detail_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Detail head output shape mismatch"

    print("Model architecture verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Fast Mode)...")

    # This will run for 2 epochs on the subset using seed 42
    run_training()

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(Config.WORKING_DIR, "model_seed_42.pth")
    if os.path.exists(expected_checkpoint):
        print(f"Checkpoint successfully created at: {expected_checkpoint}")
    else:
        raise FileNotFoundError(
            f"Training failed to produce checkpoint: {expected_checkpoint}"
        )

    # ------------------------------------------------------------------------
    # 5. Inference and Submission Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    # This uses the trained model to predict on the test subset
    generate_submission()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at: {Config.SUBMISSION_PATH}"
        )

    # Validate content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Assertions
    # Since we used a debug subset, the submission should have rows equal to the subset size
    # (or slightly less if the subset logic slices the test set exactly)
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission columns mismatch"
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    # Check probability range
    preds = df_sub["has_cactus"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Submission verification passed.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
