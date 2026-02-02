import os
import sys
import torch
import pandas as pd
import numpy as np

# Import classes and functions from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import WideResNeXt
from library.train import train_model, generate_submission
from library.utils import seed_everything


def run_pipeline_demo():
    print("=== Cactus Classification Pipeline Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override
    # --------------------------------------------------------------------------
    # We modify the Config class directly to optimize for a fast demonstration run.
    print("\n[1] Configuring environment for fast execution...")

    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 32  # Reasonable batch size
    Config.DEBUG = True  # Enable debug mode to use a subset of data
    Config.DEBUG_SAMPLE_SIZE = 128  # Use only 128 images for train/val/test
    Config.SEEDS = [42]  # Run only one seed

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG} (Samples: {Config.DEBUG_SAMPLE_SIZE})")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\n[2] Initializing DataLoaders...")

    # This will process images from metadata and cache them as .npy files in WORKING_DIR
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Training Data
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions to verify data integrity
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            32,
            32,
        ), "Train image batch shape mismatch"
        assert labels.shape == (Config.BATCH_SIZE,), "Train label batch shape mismatch"
        assert images.dtype == torch.float32, "Images should be float32 tensor"
        assert labels.dtype == torch.float32, "Labels should be float32 tensor"
        print("-> Train Loader verification passed.")
    except StopIteration:
        raise AssertionError(
            "Train loader yielded no batches. Check DEBUG_SAMPLE_SIZE vs BATCH_SIZE."
        )

    # Verify Test Data
    try:
        test_images, _ = next(iter(test_loader))
        print(f"Test Batch - Images: {test_images.shape}")
        assert test_images.shape[1:] == (3, 32, 32), "Test image dimensions mismatch"
        print("-> Test Loader verification passed.")
    except StopIteration:
        raise AssertionError("Test loader yielded no batches.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = WideResNeXt()
    model.to(Config.DEVICE)

    # Create a dummy input tensor matching the expected input shape
    dummy_input = torch.randn(2, 3, 32, 32).to(Config.DEVICE)

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assert output shape is (Batch_Size, Num_Classes) -> (2, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("-> Model forward pass verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Execution
    # --------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Seed 42)...")

    # train_model runs the loop, validation, and saves the best model to disk
    best_auc = train_model(seed=42)

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")

    # Verify that the model checkpoint was saved
    model_path = os.path.join(Config.WORKING_DIR, "model_seed_42.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    print(f"-> Model checkpoint verified at: {model_path}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    # generate_submission loads the saved model(s), runs TTA, and saves CSV
    generate_submission()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File: {Config.SUBMISSION_PATH}")
    print(f"Shape: {df_sub.shape}")
    print(df_sub.head())

    # Validate submission content
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch"
    assert len(df_sub) == len(test_loader.dataset), "Submission row count mismatch"

    # Validate probability range
    preds = df_sub["has_cactus"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]"

    print("-> Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
