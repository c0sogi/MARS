import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import CactusNet, generate_submission
from library.train import fit_model


def run_demo():
    print("============================================================")
    print("   Cactus Classification Pipeline: Demonstration & Verify   ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Use a separate directory for demo outputs to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    Config.setup()

    # Override hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 100  # Train/Test on just 100 images
    Config.SEEDS = [42]  # Single seed
    Config.BATCH_SIZE = 16  # Small batch size

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")

    # Set seed
    seed_everything(42)
    device = Config.DEVICE

    # ------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------
    print("\n[2] Verifying Data Loading and Caching...")

    # This will trigger _load_and_cache_data, creating .npy files in the demo dir
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Verify dataset sizes
    assert (
        len(train_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_loader.dataset)}"
    assert (
        len(test_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(test_loader.dataset)}"

    # Verify batch structure
    images, labels = next(iter(train_loader))

    # Expected shape: (Batch, Channels, Height, Width) -> (16, 3, 32, 32)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Image batch shape mismatch. Got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label batch shape mismatch. Got {labels.shape}"

    print("    Data Loaders initialized and verified successfully.")

    # ------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[3] Verifying CactusNet Architecture...")

    model = CactusNet(num_classes=1).to(device)

    # Create dummy input
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Perform forward pass
    try:
        output = model(dummy_input)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")

    # Verify output shape (Batch, Num_Classes)
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"

    print("    Model forward pass successful. Output shape verified.")

    # ------------------------------------------------------------------
    # 4. Training Loop & Checkpointing Verification
    # ------------------------------------------------------------------
    print("\n[4] Verifying Training Loop and Checkpointing...")

    # Run the fit function (trains for 1 epoch on 100 samples)
    fit_model(
        seed=42,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
    )

    # Verify checkpoint file existence
    expected_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "model_seed_42.pth")
    assert os.path.exists(
        expected_ckpt_path
    ), f"Checkpoint file not found at {expected_ckpt_path}"

    # Verify checkpoint content integrity
    ckpt = torch.load(expected_ckpt_path, map_location=device)
    required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "auc"]
    for key in required_keys:
        assert key in ckpt, f"Checkpoint missing key: {key}"

    print(
        f"    Training completed. Checkpoint saved and verified at: {expected_ckpt_path}"
    )

    # ------------------------------------------------------------------
    # 5. Submission Generation Verification
    # ------------------------------------------------------------------
    print("\n[5] Verifying Submission Generation (Inference + TTA)...")

    # Generate submission using the trained checkpoint
    generate_submission(test_loader, device)

    # Verify file existence
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Verify file content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check dimensions (should match debug sample size)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check columns
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission columns mismatch. Expected 'id' and 'has_cactus'."

    # Check value types
    assert pd.api.types.is_float_dtype(
        df_sub["has_cactus"]
    ), "Prediction column 'has_cactus' is not of float type."

    print(f"    Submission generated and verified at: {Config.SUBMISSION_FILE}")
    print("\n============================================================")
    print("   Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
