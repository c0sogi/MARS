import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_dataloaders
from library.model import HybridLinearProbe
from library.engine import run_training


def main():
    print("=" * 50)
    print("Starting Melanoma Detection Library Demo")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for demo
    Config.EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directory is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions...")

    # Test Reproducibility
    seed_everything(42)
    rand_a = torch.rand(5)
    seed_everything(42)
    rand_b = torch.rand(5)

    assert torch.equal(
        rand_a, rand_b
    ), "seed_everything failed to ensure reproducibility"
    print("  Reproducibility check passed.")

    # Test Checkpointing
    dummy_state = {"model_state": [1, 2, 3]}
    ckpt_dir = os.path.join(Config.WORKING_DIR, "test_ckpt")
    save_checkpoint(dummy_state, is_best=True, checkpoint_dir=ckpt_dir)

    assert os.path.exists(
        os.path.join(ckpt_dir, "checkpoint.pth")
    ), "Checkpoint file not saved"
    assert os.path.exists(
        os.path.join(ckpt_dir, "model_best.pth")
    ), "Best model copy not saved"
    print("  Checkpointing check passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoaders
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Dataset and DataLoaders...")

    # Force re-computation of metadata (load_cached_data=False) to test preprocessing pipeline
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")

    # Fetch one batch
    images, meta, targets = next(iter(train_loader))

    # Verify Shapes
    print(f"  Image Batch Shape: {images.shape}")
    print(f"  Meta Batch Shape: {meta.shape}")
    print(f"  Target Batch Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Image Shape"
    assert len(targets.shape) == 1, "Incorrect Target Shape"
    assert meta.shape[0] == Config.BATCH_SIZE, "Incorrect Metadata Batch Size"

    # Verify Metadata Dimension (needed for model init)
    meta_dim = meta.shape[1]
    print(f"  Metadata Feature Dimension: {meta_dim}")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    model = HybridLinearProbe(meta_dim=meta_dim)
    model.to(Config.DEVICE)

    # Check if backbone is frozen
    backbone_grad = any(p.requires_grad for p in model.features.parameters())
    head_grad = any(p.requires_grad for p in model.head.parameters())

    assert not backbone_grad, "Backbone should be frozen (requires_grad=False)"
    assert head_grad, "Head should be trainable (requires_grad=True)"
    print("  Backbone freezing check passed.")

    # Test Forward Pass
    images = images.to(Config.DEVICE)
    meta = meta.to(Config.DEVICE)

    logits = model(images, meta)
    print(f"  Output Logits Shape: {logits.shape}")

    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("  Forward pass check passed.")

    # -------------------------------------------------------------------------
    # 5. Execute Training Engine
    # -------------------------------------------------------------------------
    print("\n[Step 5] Executing Full Training Cycle (Engine)...")

    # We use the engine's run_training function which handles the loop, validation, and prediction
    # It will use the cached metadata generated in Step 3
    run_training(epochs=Config.EPOCHS, patience=1)

    # -------------------------------------------------------------------------
    # 6. Verify Submission
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Rows: {len(df_sub)}")
    print(f"  Columns: {list(df_sub.columns)}")
    print(f"  First 3 rows:\n{df_sub.head(3)}")

    assert "image_name" in df_sub.columns, "Missing 'image_name' column"
    assert "target" in df_sub.columns, "Missing 'target' column"

    # Verify values are probabilities
    targets = df_sub["target"].values
    assert np.all(
        (targets >= 0) & (targets <= 1)
    ), "Predictions are not valid probabilities (0-1)"

    # Verify length matches debug sample size (Config.DEBUG_SAMPLE_SIZE)
    # Note: The test loader drops samples if batch_size doesn't divide evenly depending on drop_last settings.
    # However, test loader usually has drop_last=False.
    # In debug mode, we took the first N samples.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"

    print("  Submission format check passed.")

    print("\n" + "=" * 50)
    print("Demo Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    main()
