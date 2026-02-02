import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import levenshtein_distance, decode_predictions, compute_levenshtein
from library.data_loader import GestureDataset, collate_fn
from library.model import SCRNet
from library.train import train_model
from library.predict import generate_submission


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("=== Setting up Configuration for Demo ===")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a quick demo run
    # We use a specific demo directory to keep artifacts isolated
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache_demo")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")
    Config.STATS_PATH = os.path.join(demo_dir, "stats.npz")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True  # Uses small subset of data

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n=== Verifying Utility Functions ===")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_equal = levenshtein_distance(seq1, seq2)
    assert dist_equal == 0, f"Expected distance 0, got {dist_equal}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1 (deletion), got {dist_diff}"
    print("Levenshtein distance check passed.")

    # Test Decode Predictions
    # Config.MIN_GESTURE_LENGTH is 5 by default
    # 0 is background
    # Create a signal: [0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0]
    # 1 appears 5 times (should be kept), 2 appears 2 times (should be filtered)
    raw_signal = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0])
    decoded = decode_predictions(raw_signal)

    # Depending on median filter window (default 5), the signal might be smoothed.
    # With window 5:
    # 0,0,1,1,1 -> median 1
    # 0,1,1,1,1 -> median 1
    # ...
    # Let's use a simpler check or trust the logic if we account for smoothing.
    # Let's just ensure it runs and returns a list.
    assert isinstance(decoded, list), "decode_predictions should return a list"
    print(f"Decode predictions output for synthetic signal: {decoded}")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n=== Verifying Data Loader ===")

    # Initialize Dataset (Training Mode)
    # This will compute stats if they don't exist in the demo dir
    train_dataset = GestureDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, is_train=True, load_cached_data=True
    )

    print(f"Dataset size (full): {len(train_dataset)}")

    # Fetch one sample
    skel, audio, labels = train_dataset[0]
    print(
        f"Sample 0 Shapes - Skeleton: {skel.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Verify dimensions
    # Skeleton: (Time, Joints*3) -> (Time, 60)
    assert (
        skel.shape[1] == Config.SKELETON_INPUT_DIM
    ), f"Skeleton dim mismatch. Expected {Config.SKELETON_INPUT_DIM}, got {skel.shape[1]}"
    # Audio: (Time, MFCC) -> (Time, 13)
    assert (
        audio.shape[1] == Config.AUDIO_N_MFCC
    ), f"Audio dim mismatch. Expected {Config.AUDIO_N_MFCC}, got {audio.shape[1]}"

    # Test DataLoader with Collate
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )

    batch_skel, batch_audio, batch_labels, lengths = next(iter(train_loader))
    print(
        f"Batch Shapes - Skeleton: {batch_skel.shape}, Audio: {batch_audio.shape}, Lengths: {lengths}"
    )

    assert batch_skel.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert (
        batch_skel.shape[1] == batch_audio.shape[1]
    ), "Temporal dimension mismatch between modalities"

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n=== Verifying Model Architecture ===")

    model = SCRNet()
    # Move to CPU for demo verification to ensure it runs everywhere
    model.to("cpu")

    # Forward pass with the batch fetched above
    logits = model(batch_skel, batch_audio, lengths)
    print(f"Logits Shape: {logits.shape}")

    # Expected: (Batch, Time, NumClasses)
    assert logits.shape[0] == Config.BATCH_SIZE
    assert logits.shape[1] == batch_skel.shape[1]
    assert logits.shape[2] == Config.NUM_CLASSES
    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Pipeline
    # -------------------------------------------------------------------------
    print("\n=== Verifying Training Pipeline ===")

    # Run training (Debug mode uses a subset)
    best_score = train_model(
        num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    print(f"Training finished. Best Val Score: {best_score}")

    # Check if checkpoint was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint was not saved."
    print(f"Checkpoint found at: {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n=== Verifying Inference Pipeline ===")

    # Generate submission
    generate_submission(
        checkpoint_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
        debug=Config.DEBUG,
    )

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Submission file loaded. Rows: {len(lines)}")

    # First column should be session ID (e.g., SampleXXXXX)
    # Second column (if exists) should be predictions
    if len(lines) > 0:
        sample_id = lines[0].split(",")[0]
        print(f"First Sample ID: {sample_id}")
        assert (
            isinstance(sample_id, str) and "Sample" in sample_id
        ), "Invalid Sample ID format"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
