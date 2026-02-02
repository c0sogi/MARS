import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    rle_decode,
    median_filter,
    generate_submission,
)
from library.data_loader import get_dataloaders, GestureDataset
from library.model import BS_MPII
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration of Gesture Recognition Pipeline ===")

    # 1. Setup & Configuration Override for Speed
    # We override the Config class attributes to run a fast, minimal demo.
    print("\n[1] Configuring environment for fast execution...")

    Config.DEBUG = True  # Uses a small subset (head(10)) of data
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Ensure clean directories
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create dummy metadata if original files are missing/empty for the demo context
    # (The environment usually has them, but we ensure the paths match Config)
    # We rely on the existing ./metadata files as per instructions.

    set_seed(Config.SEED)
    print("Configuration updated. Debug mode: ON.")

    # 2. Validate Utility Functions
    print("\n[2] Validating Utility Functions...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical seqs should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance should be 1 (deletion), got {dist_diff}"
    print("Levenshtein distance logic verified.")

    # Test RLE Decode
    # Sequence: 1, 1, 1, 0, 0, 2, 2, 2, 2, 2 (Background=0)
    # Min length = 3
    raw_preds = [1, 1, 1, 0, 0, 2, 2, 2, 2, 2]
    decoded = rle_decode(raw_preds, min_length=3, background_class=0)
    # Expect [1, 2]
    assert decoded == [1, 2], f"RLE Decode failed. Expected [1, 2], got {decoded}"

    # Test filtering short gestures
    raw_preds_short = [1, 1, 0, 0, 2, 2, 2]  # 1 is length 2 (<3), should be skipped
    decoded_short = rle_decode(raw_preds_short, min_length=3, background_class=0)
    assert decoded_short == [
        2
    ], f"RLE Decode short filter failed. Expected [2], got {decoded_short}"
    print("RLE Decode logic verified.")

    # 3. Validate Data Loading
    print("\n[3] Validating Data Loading...")

    # Initialize Loaders (this triggers stats computation if not present)
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch
    batch = next(iter(train_loader))
    assert batch is not None, "DataLoader returned None."

    # Verify Batch Structure
    required_keys = [
        "skeleton",
        "audio",
        "labels",
        "boundaries",
        "lengths",
        "sample_ids",
    ]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    skel = batch["skeleton"]
    audio = batch["audio"]
    labels = batch["labels"]
    lengths = batch["lengths"]

    # Check dimensions
    # Skeleton: (B, T, J, 3)
    assert skel.dim() == 4, f"Skeleton dim mismatch. Expected 4, got {skel.dim()}"
    assert skel.shape[2] == Config.SKELETON_JOINTS, "Skeleton joints mismatch"
    assert skel.shape[3] == 3, "Skeleton channels mismatch"

    # Audio: (B, T, C)
    assert audio.dim() == 3, f"Audio dim mismatch. Expected 3, got {audio.dim()}"
    assert audio.shape[2] == Config.AUDIO_N_MFCC, "Audio MFCC mismatch"

    # Labels: (B, T)
    assert labels.dim() == 2, f"Labels dim mismatch. Expected 2, got {labels.dim()}"

    print(f"Batch shapes verified. Skeleton: {skel.shape}, Audio: {audio.shape}")

    # 4. Validate Model Architecture
    print("\n[4] Validating Model Architecture...")

    model = BS_MPII().to(Config.DEVICE)

    # Move batch to device
    skel_dev = skel.to(Config.DEVICE)
    audio_dev = audio.to(Config.DEVICE)
    lengths_dev = lengths.to(Config.DEVICE)

    # Forward Pass
    outputs = model(skel_dev, audio_dev, lengths_dev)

    assert "class_logits" in outputs, "Model output missing class_logits"
    assert "boundary_logits" in outputs, "Model output missing boundary_logits"

    class_logits = outputs["class_logits"]
    boundary_logits = outputs["boundary_logits"]

    # Check Output Shapes
    # class_logits: (B, T, NumClasses)
    assert class_logits.shape[0] == skel.shape[0]
    assert class_logits.shape[1] == skel.shape[1]
    assert class_logits.shape[2] == Config.NUM_CLASSES

    # boundary_logits: (B, T)
    assert boundary_logits.shape[0] == skel.shape[0]
    assert boundary_logits.shape[1] == skel.shape[1]

    print("Model forward pass successful. Output shapes verified.")

    # 5. Validate Training Loop (Trainer)
    print("\n[5] Validating Trainer (1 Epoch)...")

    trainer = Trainer()

    # We will run fit(). Since Config.NUM_EPOCHS=1 and DEBUG=True, this should be fast.
    trainer.fit()

    # Check if checkpoint was created
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Checkpoint successfully created at {Config.BEST_MODEL_PATH}")
    else:
        # It's possible validation score didn't improve from infinity (unlikely) or 0.
        # But Trainer initializes best_val_score to inf, so any valid score < inf saves.
        # If validation set is empty or fails, it might not save.
        # In debug mode with small data, it should save.
        print(
            "Warning: No checkpoint found. This might happen if validation score was NaN or Inf."
        )

    # 6. Validate Submission Generation
    print("\n[6] Validating Submission Generation...")

    # Create dummy predictions dictionary
    dummy_preds = {
        "Sample001": [1, 2, 3],
        "Sample002": [4, 5],
        "Sample003": [],  # No gestures
    }

    generate_submission(dummy_preds, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Read and check content
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    # Expect 3 lines
    assert len(lines) == 3, f"Expected 3 lines in submission, got {len(lines)}"

    # Check format
    # Sample001,1,2,3
    assert (
        "Sample001,1,2,3" in lines[0]
        or "Sample001,1,2,3" in lines[1]
        or "Sample001,1,2,3" in lines[2]
    )
    # Sample003
    assert "Sample003\n" in lines or "Sample003" in lines[-1]

    print(f"Submission file verified at {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
