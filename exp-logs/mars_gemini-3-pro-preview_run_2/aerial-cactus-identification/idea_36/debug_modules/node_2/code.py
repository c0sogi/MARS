import os
import sys
import numpy as np
import torch
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import UltraWideECARepNeXt, RepNeXtUnit
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override for Speed
    print("\n[Step 1] Configuring environment...")
    set_seed(42)

    # Override Config for a quick demo run
    Config.EPOCHS = 1
    Config.SEEDS = [42]  # Single seed for speed
    Config.BATCH_SIZE = 8

    # Ensure working directories exist (handled by Config.setup(), but good to confirm)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print("Configuration updated for rapid execution.")

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    # Use num_workers=0 to avoid multiprocessing overhead in this short script
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch a single batch from train_loader
    images, labels, ids = next(iter(train_loader))

    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for data integrity
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert images.shape[2] == 32 and images.shape[3] == 32, "Images must be 32x32"
    assert labels.shape[0] == Config.BATCH_SIZE, "Labels batch size mismatch"
    assert len(ids) == Config.BATCH_SIZE, "IDs batch size mismatch"
    print("Data loading verification passed.")

    # 3. Model Architecture and Re-parameterization Verification
    print("\n[Step 3] Verifying Model and Re-parameterization...")
    device = torch.device(
        "cpu"
    )  # Use CPU for simple logic checks to avoid GPU overhead initialization if not needed
    model = UltraWideECARepNeXt().to(device)

    # Check Forward Pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    output = model(dummy_input)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model forward pass successful.")

    # Inspect internal structure before deploy
    # We look at the first block of stage 1
    block = model.stage1[0]
    assert isinstance(block, RepNeXtUnit), "Expected RepNeXtUnit block"
    assert hasattr(
        block, "branch_3x3_conv"
    ), "Block should have branch_3x3_conv before deploy"
    assert not block.deploy, "Block deploy flag should be False initially"

    # Switch to deploy (Fuse weights)
    print("Switching model to deploy mode (fusing weights)...")
    model.switch_to_deploy()

    # Inspect internal structure after deploy
    assert block.deploy, "Block deploy flag should be True after switch"
    assert hasattr(block, "fused_conv"), "Block should have fused_conv after deploy"
    assert not hasattr(
        block, "branch_3x3_conv"
    ), "branch_3x3_conv should be removed after deploy"

    # Check Forward Pass after fusion
    output_fused = model(dummy_input)
    assert output_fused.shape == (2, 1), "Fused model output shape mismatch"
    print("Model re-parameterization verification passed.")

    # 4. Integration Test: Full Training and Inference Pipeline
    print("\n[Step 4] Running Full Pipeline (Debug Mode)...")
    # run_training with debug=True truncates data to 100 samples and runs for 2 epochs (hardcoded in train.py for debug)
    # We will rely on the debug flag to ensure this finishes quickly.

    # Note: run_training prints its own logs.
    run_training(epochs=1, seeds=[42], batch_size=16, debug=True)

    print("Pipeline execution completed.")

    # 5. Submission Verification
    print("\n[Step 5] Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")

    # Assertions for submission format
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    # In debug mode, run_training truncates test set to 100 samples.
    # We verify that we have exactly 100 predictions.
    assert (
        len(df_sub) == 100
    ), f"Expected 100 predictions in debug mode, got {len(df_sub)}"

    # Check probability range
    probs = df_sub["has_cactus"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
