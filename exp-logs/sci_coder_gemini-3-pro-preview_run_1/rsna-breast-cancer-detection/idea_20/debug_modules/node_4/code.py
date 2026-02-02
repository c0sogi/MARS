import os
import sys
import shutil
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, probabilistic_f1, get_logger
from library.data import get_dataloaders
from library.model import SiameseEfficientNet
from library.train_eval import train_one_epoch, evaluate, inference


def cleanup_resources():
    """
    Releases memory and clears GPU cache to prevent OOM.
    Cite debug_lesson_18: Purge System Tracebacks to Release Zombie GPU Memory.
    Cite debug_lesson_27: Explicitly Release GPU Resources Before Invoking Independent Sub-Routines.
    """
    # Clear system traceback to release zombie references
    if hasattr(sys, "last_traceback"):
        sys.last_traceback = None

    gc.collect()
    torch.cuda.empty_cache()


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("Setting up configuration for fast demonstration...")

    # Override Config for speed and resources
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.IMG_SIZE = (256, 256)  # Reduced size for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Use a specific cache directory for this demo to avoid conflicts
    Config.IDEA_CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    if os.path.exists(Config.IDEA_CACHE_DIR):
        shutil.rmtree(Config.IDEA_CACHE_DIR)
    os.makedirs(Config.IDEA_CACHE_DIR, exist_ok=True)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\nVerifying Utility Functions...")

    # Test Probabilistic F1
    y_true_dummy = np.array([1, 0, 1, 0])
    y_pred_dummy = np.array([0.9, 0.1, 0.8, 0.2])
    pf1 = probabilistic_f1(y_true_dummy, y_pred_dummy)
    print(f"  Dummy pF1 Score: {pf1:.4f}")

    # Assert reasonable bounds
    assert 0.0 <= pf1 <= 1.0, "pF1 score out of bounds"

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\nVerifying Data Pipeline...")

    # Get DataLoaders (this triggers metadata processing)
    # forcing load_cached_data=False to test processing logic
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs, targets = batch

    # Verify Inputs
    assert "target" in inputs, "Input dict missing 'target' key"
    assert "contra" in inputs, "Input dict missing 'contra' key"

    img_t = inputs["target"]
    img_c = inputs["contra"]

    print(f"  Input Batch Shape: {img_t.shape}")
    print(f"  Target Labels Shape: {targets.shape}")

    # Assert Shapes: (B, C, H, W)
    # C=3 (Image, Age, Implant)
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        img_t.shape == expected_shape
    ), f"Expected {expected_shape}, got {img_t.shape}"
    assert (
        img_c.shape == expected_shape
    ), f"Expected {expected_shape}, got {img_c.shape}"

    # -------------------------------------------------------------------------
    # 4. Verify Model & Forward Pass
    # -------------------------------------------------------------------------
    print("\nVerifying Model & Forward Pass...")

    # Explicitly delete batch to free memory before model allocation
    del batch, inputs, targets, img_t, img_c
    cleanup_resources()

    model = SiameseEfficientNet().to(device)

    # Re-fetch a batch for forward pass verification
    batch = next(iter(train_loader))
    inputs, targets = batch

    # Move batch to device
    img_t = inputs["target"].to(device)
    img_c = inputs["contra"].to(device)

    # Forward
    logits = model(img_t, img_c)

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch"

    # -------------------------------------------------------------------------
    # 5. Verify Training Step
    # -------------------------------------------------------------------------
    print("\nVerifying Training Step...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch (on the tiny subset)
    train_loss, train_pf1 = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train pF1:  {train_pf1:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 6. Verify Evaluation Step
    # -------------------------------------------------------------------------
    print("\nVerifying Evaluation Step...")

    val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val pF1:  {val_pf1:.4f}")

    # -------------------------------------------------------------------------
    # 7. Verify Inference & Submission
    # -------------------------------------------------------------------------
    print("\nVerifying Inference & Submission...")

    # Run inference
    inference(model, test_loader, device)

    # Check if file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Rows: {len(df_sub)}")
    print("  Submission Head:")
    print(df_sub.head())

    required_cols = ["prediction_id", "cancer"]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission"

    # Check probability range
    probs = df_sub["cancer"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
