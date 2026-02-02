import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_loaders, GestureDataset
from library.model import SGCRCN
from library.trainer import Trainer


def run_demo():
    print("=== Starting SG-CRCN Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples per split
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 2  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"

    # Update paths based on new working directory
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = Config.get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Initialize loaders (force processing from scratch to verify logic)
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify dataset size
    print(f"    Train dataset size: {len(train_loader.dataset)}")
    assert (
        len(train_loader.dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} train samples, got {len(train_loader.dataset)}"

    # Verify batch structure
    batch = next(iter(train_loader))
    features = batch["features"]
    target_cls = batch["target_cls"]
    mask = batch["mask"]

    print(f"    Batch Features Shape: {features.shape} (B, T, D)")
    print(f"    Batch Targets Shape: {target_cls.shape} (B, T)")
    print(f"    Batch Mask Shape: {mask.shape} (B, T)")

    # Assertions
    assert features.ndim == 3, "Features should be 3D (Batch, Time, Dim)"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {features.shape[2]}"
    assert target_cls.shape == mask.shape, "Target and Mask shapes must match"

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = SGCRCN().to(device)

    # Move batch to device
    b_features = features.to(device)
    b_mask = mask.to(device)

    # Forward pass
    outputs = model(b_features, b_mask)

    # Check outputs
    required_keys = [
        "stage1_cls",
        "stage1_bnd",
        "stage2_cls",
        "stage2_bnd",
        "stage3_cls",
        "stage3_bnd",
    ]
    for key in required_keys:
        assert key in outputs, f"Model output missing key: {key}"
        assert (
            outputs[key].shape[0] == Config.BATCH_SIZE
        ), f"Batch size mismatch in {key}"

    print("    Forward pass successful. Output keys verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Training Loop...")

    trainer = Trainer(device, train_loader, val_loader, test_loader)

    # Run training
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Check if checkpoint was saved
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"    Training complete. Checkpoint found at {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference...")

    trainer.predict()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content format
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"    Generated {len(lines)} lines in submission file.")
        assert (
            len(lines) == Config.DEBUG_SUBSET_SIZE
        ), f"Expected {Config.DEBUG_SUBSET_SIZE} predictions, got {len(lines)}"

        # Check first line format: SessionID,Label1,Label2...
        parts = lines[0].strip().split(",")
        assert len(parts) >= 1, "Invalid submission format"
        print(f"    Sample prediction line: {lines[0].strip()}")

    # -------------------------------------------------------------------------
    # 6. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Metric Calculation...")

    # Perfect match
    score_perfect = compute_levenshtein([[1, 2, 3]], [[1, 2, 3]])
    assert (
        score_perfect == 0.0
    ), f"Perfect match should have 0 error, got {score_perfect}"

    # Complete mismatch
    score_mismatch = compute_levenshtein([[1]], [[2]])
    assert score_mismatch > 0.0, "Mismatch should have error > 0"

    # Empty prediction vs Target
    score_empty = compute_levenshtein([[]], [[1, 2]])
    # Levenshtein distance is len(target) = 2. Total target len = 2. Score = 2/2 = 1.0
    assert (
        abs(score_empty - 1.0) < 1e-6
    ), f"Expected error 1.0 for empty pred, got {score_empty}"

    print("    Metric logic verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
