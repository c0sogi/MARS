import os
import sys
import torch
import pandas as pd
import numpy as np

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import ArcFaceResNet
from library.train import run_training


def main():
    print("=== Starting Plant Species Classification Demo ===")

    # 1. Setup and Configuration
    # We modify the batch size to be small for this demonstration to ensure
    # the data loaders work correctly with the small debug dataset size.
    Config.setup()
    Config.BATCH_SIZE = 16
    DEBUG_SAMPLE_SIZE = 64

    print(f"Configuration:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Debug Sample Size: {DEBUG_SAMPLE_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    set_seed(Config.SEED)

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loaders ---")
    # We set load_cached_data=False to force the sampler to recalculate weights
    # based on the small debug subset rather than looking for a full-dataset cache.
    dataloaders = get_dataloaders(
        load_cached_data=False, debug_sample_size=DEBUG_SAMPLE_SIZE
    )
    train_loader = dataloaders["train"]

    # Fetch a single batch to verify shapes
    try:
        images, labels = next(iter(train_loader))
        print(f"  Train Batch Images Shape: {images.shape}")
        print(f"  Train Batch Labels Shape: {labels.shape}")

        # Assertions to ensure data is loaded correctly
        assert (
            images.shape[0] == Config.BATCH_SIZE
        ), f"Expected batch size {Config.BATCH_SIZE}, got {images.shape[0]}"
        assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
        assert (
            images.shape[2] == Config.IMAGE_SIZE
        ), f"Expected height {Config.IMAGE_SIZE}, got {images.shape[2]}"
        assert (
            images.shape[3] == Config.IMAGE_SIZE
        ), f"Expected width {Config.IMAGE_SIZE}, got {images.shape[3]}"
        assert (
            labels.shape[0] == Config.BATCH_SIZE
        ), f"Expected label batch size {Config.BATCH_SIZE}, got {labels.shape[0]}"
        print("  Data Loader shapes verified successfully.")
    except StopIteration:
        raise AssertionError(
            "Train loader is empty. Ensure debug_sample_size >= batch_size when drop_last=True."
        )

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    model = ArcFaceResNet().to(Config.DEVICE)

    # Create dummy inputs
    dummy_images = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)
    dummy_labels = torch.randint(0, Config.NUM_CLASSES, (Config.BATCH_SIZE,)).to(
        Config.DEVICE
    )

    # Test Inference Mode (No labels provided)
    print("  Testing Inference Forward Pass...")
    with torch.no_grad():
        output_inference = model(dummy_images)

    assert output_inference.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Inference output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output_inference.shape}"
    print("  Inference pass successful.")

    # Test Training Mode (Labels provided, ArcFace margin applied)
    print("  Testing Training Forward Pass...")
    output_train = model(dummy_images, dummy_labels)
    assert output_train.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Training output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output_train.shape}"
    print("  Training pass successful.")

    # 4. Training Pipeline Execution
    print("\n--- Running Training Pipeline (1 Epoch, Subset) ---")
    # This function handles the training loop, validation, model saving, and submission generation.
    run_training(
        debug_sample_size=DEBUG_SAMPLE_SIZE, num_epochs=1, load_cached_data=False
    )

    # 5. Submission Verification
    print("\n--- Verifying Submission Output ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission file loaded from {Config.SUBMISSION_PATH}")
    print(f"  Rows: {len(submission_df)}")
    print("  Head:")
    print(submission_df.head())

    # Validate submission format
    assert "Id" in submission_df.columns, "Submission missing 'Id' column"
    assert "Predicted" in submission_df.columns, "Submission missing 'Predicted' column"

    # Since we used a debug subset for the test set as well, the submission size should match
    assert (
        len(submission_df) == DEBUG_SAMPLE_SIZE
    ), f"Submission size {len(submission_df)} does not match debug sample size {DEBUG_SAMPLE_SIZE}"

    print("  Submission format verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
