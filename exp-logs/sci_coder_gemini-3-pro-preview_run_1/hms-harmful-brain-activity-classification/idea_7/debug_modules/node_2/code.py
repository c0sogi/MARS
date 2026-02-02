import os
import sys
import shutil
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger, kl_divergence_score
from library.data import get_dataloaders
from library.model import OffsetGuidedDualStreamModel
from library.train import train_one_epoch, validate
from library.inference import run_inference


def run_demonstration():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Setting up Configuration for Demo...")

    # Modify Config for a fast, minimal execution
    Config.DEBUG = True
    Config.TRAIN_SUBSET_SIZE = 32  # Small subset for training
    Config.VAL_SUBSET_SIZE = 16  # Small subset for validation
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Clear cache to ensure data processing logic is applied (Cite debug_lesson_10)
    if os.path.exists(Config.CACHE_DIR):
        print(f"Clearing stale cache at {Config.CACHE_DIR}...")
        shutil.rmtree(Config.CACHE_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Setup logger
    logger = get_logger("demo_logger")
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # =========================================================================
    # 2. Data Loading & Verification
    # =========================================================================
    print("\n>>> Initializing DataLoaders...")

    # get_dataloaders handles caching and dataset creation
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=True
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    spec, eeg, guidance, targets = next(iter(train_loader))

    # Move to device
    spec = spec.to(device)
    eeg = eeg.to(device)
    guidance = guidance.to(device)
    targets = targets.to(device)

    print(f"\n[Batch Verification]")
    print(f"Spectrogram Shape: {spec.shape} (Expected: [B, 3, 512, 512])")
    print(f"EEG Shape: {eeg.shape} (Expected: [B, 20, 5000])")
    print(f"Guidance Shape: {guidance.shape} (Expected: [B])")
    print(f"Targets Shape: {targets.shape} (Expected: [B, 6])")

    # Assertions
    assert (
        spec.ndim == 4 and spec.shape[1] == 3
    ), "Spectrogram tensor has incorrect dimensions."
    assert eeg.ndim == 3 and eeg.shape[1] == 20, "EEG tensor has incorrect dimensions."
    assert targets.shape[1] == 6, "Targets must have 6 classes."

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n>>> Initializing Model...")
    model = OffsetGuidedDualStreamModel(Config).to(device)

    print(">>> Running Forward Pass...")
    logits = model(spec, eeg, guidance)

    print(f"Logits Shape: {logits.shape}")

    # Check output shape
    assert logits.shape == (Config.BATCH_SIZE, 6), "Model output shape mismatch."
    # Check that outputs are finite (no NaNs)
    assert torch.isfinite(logits).all(), "Model produced NaN or Inf values."

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("\n>>> Demonstrating Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.MAX_LR)
    scheduler = None  # Skipping scheduler for this simple demo step

    # Run one epoch of training
    train_loss = train_one_epoch(
        epoch=1,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
    )

    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    print(">>> Demonstrating Validation Step...")
    val_loss = validate(model, val_loader, device, logger)
    print(f"Validation Loss: {val_loss:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN."

    # Save a dummy checkpoint for inference to use
    dummy_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), dummy_ckpt_path)
    print(f"Saved dummy checkpoint to {dummy_ckpt_path}")

    # =========================================================================
    # 5. Inference Pipeline Demonstration
    # =========================================================================
    print("\n>>> Demonstrating Inference Pipeline...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Run inference using the helper function from library.inference
    # We pass the checkpoint we just saved
    submission_df = run_inference(
        checkpoint_path=dummy_ckpt_path,
        output_path=submission_path,
        debug=True,
        batch_size=Config.BATCH_SIZE,
    )

    print(f"Submission file created at: {submission_path}")

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not created."

    loaded_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {loaded_sub.shape}")
    print("First 3 rows:")
    print(loaded_sub.head(3))

    # Check columns
    expected_cols = ["eeg_id"] + Config.CLASS_NAMES
    assert list(loaded_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check probability sum constraint
    prob_sums = loaded_sub[Config.CLASS_NAMES].sum(axis=1)
    # Allow small floating point error
    assert np.allclose(prob_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0."

    print("\n>>> Demonstration Complete. All checks passed.")


if __name__ == "__main__":
    run_demonstration()
