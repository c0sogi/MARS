import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model_components import RepNeXtBlock
from library.train import run_training
from library.inference import run_inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config to use a specific demo directory and limit runtime
    DEMO_DIR = "./working/demo_execution"

    # Clean up previous run if exists to ensure a fresh start
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    # Override paths and settings
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Reduce epochs for speed (demonstration only)
    Config.EPOCHS = 1
    Config.SEEDS = [42]

    # Apply setup to create the new directories
    Config.setup()

    # Set reproducible seed
    set_seed(42)

    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n--- Verifying Utilities ---")
    # Test ROC AUC with known values
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # Expected AUC calculation:
    # Pairs: (0, 0.1) vs (1, 0.35) -> Correct
    # (0, 0.1) vs (1, 0.8) -> Correct
    # (0, 0.4) vs (1, 0.35) -> Incorrect
    # (0, 0.4) vs (1, 0.8) -> Correct
    # Total 3/4 = 0.75
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")
    assert auc == 0.75, f"ROC AUC calculation failed. Expected 0.75, got {auc}"

    # -------------------------------------------------------------------------
    # 3. Verify Model Components (Structural Re-parameterization)
    # -------------------------------------------------------------------------
    print("\n--- Verifying RepNeXtBlock Re-parameterization ---")
    # Instantiate block
    in_ch, out_ch = 64, 64
    block = RepNeXtBlock(in_ch, out_ch, groups=32, deploy=False)
    block.eval()  # Set to eval mode to freeze BN stats for valid comparison

    # Create dummy input
    x = torch.randn(2, in_ch, 32, 32)

    # 1. Forward pass with multi-branch structure
    with torch.no_grad():
        out_branch = block(x)

    # 2. Switch to deploy (fuse branches)
    block.switch_to_deploy()

    # Check structure
    assert block.deploy is True, "Block failed to switch to deploy mode"
    assert hasattr(block, "fused_conv"), "Fused convolution layer missing"
    assert not hasattr(block, "branch_3x3"), "Original branches not removed"

    # 3. Forward pass with fused structure
    with torch.no_grad():
        out_fused = block(x)

    # 4. Compare outputs
    diff = (out_branch - out_fused).abs().max().item()
    print(f"Max difference between branched and fused output: {diff:.6e}")
    # Tolerance for float32 precision
    assert diff < 1e-4, f"Re-parameterization output mismatch. Diff: {diff}"

    # -------------------------------------------------------------------------
    # 4. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Loading ---")
    # This will load metadata, cache images to .npy in WORKING_DIR, and return loaders
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Check batch structure
    images, labels = next(iter(train_loader))
    print(f"Train Batch Images Shape: {images.shape}")
    print(f"Train Batch Labels Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # -------------------------------------------------------------------------
    # 5. Execute Training (1 Epoch)
    # -------------------------------------------------------------------------
    print("\n--- Executing Training (1 Epoch) ---")
    # Train a single model instance
    # This exercises the model, optimizer, loss, and checkpointing logic
    run_training(seed=42, epochs=1, load_cached_data=True)

    # Verify checkpoint existence
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_seed_42.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint not created at {checkpoint_path}"
    print(f"Checkpoint verified: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 6. Execute Inference
    # -------------------------------------------------------------------------
    print("\n--- Executing Inference ---")
    # Run inference using the trained checkpoint
    # This exercises loading, re-parameterization, TTA, and submission generation
    run_inference(seeds=[42], load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Check submission content
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df)} rows.")
    assert len(df) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df)}"
    assert list(df.columns) == ["id", "has_cactus"], "Submission columns mismatch"

    print("\n=== All Demonstrations and Validations Passed Successfully ===")


if __name__ == "__main__":
    main()
