import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import make_dataloaders
from library.model import CGWBN
from library.layers import CBAM, DualPooling, ContextGating
from library.engine import train_fold, predict


def run_demo():
    print("=== Starting Library Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Modify Config for a fast, isolated run
    Config.DEBUG = True
    Config.MAX_SAMPLES = 32  # Small subset for speed
    Config.NUM_EPOCHS = 1  # Only 1 epoch for demonstration
    Config.BATCH_SIZE = 8
    Config.PATIENCE = 1
    Config.IDEA_ID = "demo_run"
    Config.WORK_DIR = os.path.join("./working", Config.IDEA_ID)
    Config.CACHE_FILE = os.path.join(Config.WORK_DIR, "cache", "processed_data.npz")
    Config.MODEL_CHECKPOINT_PATTERN = os.path.join(
        Config.WORK_DIR, "model_fold_{fold}.pth"
    )
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean working directories
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.CACHE_FILE), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("    Configuration updated. Debug mode: ON")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Testing Data Pipeline...")

    # Generate dataloaders
    # load_cached_data=False forces processing from scratch to verify logic
    train_loader, val_loader, test_loader = make_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["inc_angle"]
    labels = batch["label"]
    ids = batch["id"]

    print(
        f"    Sample Batch Shapes -> Image: {images.shape}, Angle: {angles.shape}, Label: {labels.shape}"
    )

    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert images.shape[2] == 75 and images.shape[3] == 75, "Images must be 75x75"
    assert angles.dim() == 2, "Incidence angles must be 2D tensors (B, 1)"
    assert labels.dim() == 2, "Labels must be 2D tensors (B, 1)"
    print("    Data Loader verification passed.")

    # ---------------------------------------------------------
    # 3. Model Logic & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    device = Config.DEVICE
    model = CGWBN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    logits = model(images, angles)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        images.size(0),
        1,
    ), "Output shape mismatch. Expected (Batch, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("    Model forward pass verification passed.")

    # ---------------------------------------------------------
    # 4. Component Unit Tests
    # ---------------------------------------------------------
    print("\n[4] Testing Custom Layers...")

    # Test DualPooling
    # Input: (B, C, H, W) -> Output: (B, 2C, H/2, W/2)
    dummy_input = torch.randn(2, 16, 10, 10).to(device)
    pool = DualPooling(kernel_size=2, stride=2).to(device)
    pool_out = pool(dummy_input)
    assert pool_out.shape == (
        2,
        32,
        5,
        5,
    ), f"DualPooling shape incorrect: {pool_out.shape}"
    print("    DualPooling: OK")

    # Test CBAM
    # Should maintain shape
    cbam = CBAM(in_planes=16).to(device)
    cbam_out = cbam(dummy_input)
    assert cbam_out.shape == dummy_input.shape, "CBAM altered output shape"
    print("    CBAM: OK")

    # Test ContextGating
    # Features: (B, 10), Context: (B, 5)
    cg = ContextGating(context_dim=5, feature_dim=10).to(device)
    feat = torch.randn(2, 10).to(device)
    ctx = torch.randn(2, 5).to(device)
    cg_out = cg(feat, ctx)
    assert cg_out.shape == feat.shape, "ContextGating altered output shape"
    print("    ContextGating: OK")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Testing Training Loop (Fold 0)...")

    # Train for 1 epoch (as configured)
    trained_model = train_fold(0, model, train_loader, val_loader, device)

    # Verify checkpoint creation
    ckpt_path = Config.MODEL_CHECKPOINT_PATTERN.format(fold=0)
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"    Training finished. Checkpoint saved to {ckpt_path}")

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Testing Inference...")

    ids, probs = predict(trained_model, test_loader, device)

    print(f"    Predictions generated: {len(ids)} IDs, {len(probs)} Probabilities")
    print(f"    First 3 IDs: {ids[:3]}")
    print(f"    First 3 Probs: {probs[:3]}")

    # Assertions
    assert len(ids) == len(probs), "Mismatch between IDs and probabilities count"
    assert len(ids) == len(
        test_loader.dataset
    ), "Prediction count matches test set size"
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of [0, 1] range"

    # Create submission file
    sub_df = pd.DataFrame({"id": ids, "is_iceberg": probs})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
