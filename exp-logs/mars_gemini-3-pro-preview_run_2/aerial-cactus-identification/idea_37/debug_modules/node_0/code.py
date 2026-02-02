import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.dataset import get_loaders
from library.model_components import RepNeXtBlock, UltraWideSERepNeXt
from library.train import train_seed, run_inference
from library.utils import set_seed


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("Configuring environment for demo run...")

    # Override Config constants to ensure the script runs quickly (Debug Mode)
    Config.EPOCHS = 1
    Config.SEEDS = [42]  # Only run one seed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 128  # Small subset for speed
    Config.BATCH_SIZE = 32  # Smaller batch size for the small subset

    # Redirect outputs to a specific demo directory to avoid clutter/conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Create the directories manually since Config usually does this at import time
    # and we just changed the paths.
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Loading ---")

    # Force reload to ensure we don't use existing cache from other runs
    # Note: get_loaders handles the DEBUG slicing logic.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Check Train Loader
    images, targets = next(iter(train_loader))
    print(f"Train Batch Shape: {images.shape}")
    print(f"Train Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect train image shape"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect train target shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # Check Test Loader
    test_images, test_ids = next(iter(test_loader))
    print(f"Test Batch Shape: {test_images.shape}")
    assert test_images.shape[1:] == (3, 32, 32), "Incorrect test image dimensions"
    assert len(test_ids) == Config.BATCH_SIZE, "Incorrect number of test IDs"

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Components
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Components ---")

    device = torch.device("cpu")  # Use CPU for simple logic verification

    # Test RepNeXtBlock
    print("Testing RepNeXtBlock...")
    in_ch, out_ch = 32, 64
    block = RepNeXtBlock(in_ch, out_ch, stride=1, groups=8, deploy=False).to(device)
    dummy_input = torch.randn(2, in_ch, 32, 32).to(device)

    # Forward pass (Training mode)
    out_train = block(dummy_input)
    assert out_train.shape == (2, out_ch, 32, 32), "RepNeXtBlock output shape mismatch"

    # Switch to deploy
    block.switch_to_deploy()
    assert block.deploy is True, "Block did not switch to deploy mode"
    assert hasattr(block, "reparam_conv"), "Fused convolution not found"
    assert not hasattr(block, "conv3x3"), "Training branches not removed"

    # Forward pass (Deploy mode)
    out_deploy = block(dummy_input)
    assert out_deploy.shape == (
        2,
        out_ch,
        32,
        32,
    ), "RepNeXtBlock deploy output shape mismatch"

    # Check numerical consistency (should be close, but maybe not identical due to float precision)
    diff = (out_train - out_deploy).abs().max().item()
    print(f"RepNeXtBlock Fusion Max Difference: {diff:.6f}")
    # We expect a small difference, usually < 1e-4 or 1e-5 depending on depth
    assert diff < 1e-3, "Fusion resulted in significant numerical deviation"

    # Test Full Model
    print("Testing UltraWideSERepNeXt...")
    model = UltraWideSERepNeXt(num_classes=1, deploy=False).to(device)
    dummy_input_model = torch.randn(2, 3, 32, 32).to(device)
    out_model = model(dummy_input_model)

    assert out_model.shape == (2, 1), "Model output shape mismatch"
    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Execute Training (Demo)
    # -------------------------------------------------------------------------
    print("\n--- Executing Training (Seed 42) ---")
    # This uses library.train.train_seed
    # It will use the Config overrides we set earlier (Epochs=1, Debug=True)
    train_seed(42)

    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "model_seed_42.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not created at {expected_checkpoint}"
    print("Training completed and checkpoint verified.")

    # -------------------------------------------------------------------------
    # 5. Execute Inference (Demo)
    # -------------------------------------------------------------------------
    print("\n--- Executing Inference ---")
    # This uses library.train.run_inference
    # It loads the checkpoint we just created and runs prediction
    run_inference()

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not created at {Config.SUBMISSION_PATH}"
    print("Inference completed and submission file verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Submission Content
    # -------------------------------------------------------------------------
    print("\n--- Verifying Submission Content ---")
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    assert "id" in df_sub.columns, "Missing 'id' column"
    assert "has_cactus" in df_sub.columns, "Missing 'has_cactus' column"

    # In debug mode, the test set is sliced to DEBUG_SAMPLE_SIZE
    # However, get_loaders logic for test set:
    # limit_test = min(len(test_imgs), Config.DEBUG_SAMPLE_SIZE)
    # So we expect exactly DEBUG_SAMPLE_SIZE rows
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"

    # Check probability range
    probs = df_sub["has_cactus"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Submission content verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
