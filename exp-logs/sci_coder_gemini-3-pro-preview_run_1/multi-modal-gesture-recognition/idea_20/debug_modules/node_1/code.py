import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, set_seed
from library.utils import (
    levenshtein_distance,
    rle_decode,
    compute_levenshtein_score,
    apply_median_filter,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import DAGINet
from library.train import run_training


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print(">>> Setting up demonstration environment...")

    # Define paths for demo
    demo_dir = "./working/demo_execution"
    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_checkpoints_dir = os.path.join(demo_dir, "checkpoints")
    demo_submission_dir = os.path.join(demo_dir, "submission")

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_checkpoints_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Load original metadata and create tiny subsets for speed
    # We use the first few samples that actually exist
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Take top 5 samples for train/val/test
    subset_train = full_train_df.head(5)
    subset_val = full_val_df.head(5)
    subset_test = full_test_df.head(5)

    # Save subsets
    demo_train_path = os.path.join(demo_metadata_dir, "train.csv")
    demo_val_path = os.path.join(demo_metadata_dir, "val.csv")
    demo_test_path = os.path.join(demo_metadata_dir, "test.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)
    subset_test.to_csv(demo_test_path, index=False)

    print(
        f"Created subset metadata: {len(subset_train)} train, {len(subset_val)} val samples."
    )

    # Patch Config at runtime to use demo paths and settings
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path
    Config.BEST_MODEL_PATH = os.path.join(demo_checkpoints_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Optimize hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.HIDDEN_SIZE = 64  # Reduce model size for demo

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Test Levenshtein Distance
    seq_a = [1, 2, 3]
    seq_b = [1, 3]  # Deletion of '2'
    dist = levenshtein_distance(seq_a, seq_b)
    assert dist == 1, f"Expected distance 1, got {dist}"
    print("Levenshtein distance logic: OK")

    # Test RLE Decode
    # Sequence: Background(0) -> Class(1) -> Background(0) -> Class(2)
    # Filter: min_duration=3
    raw_preds = np.array([0, 0, 1, 1, 1, 1, 0, 2, 2, 0, 0])
    # Class 1 has length 4 (>=3) -> Keep
    # Class 2 has length 2 (<3) -> Discard
    decoded = rle_decode(raw_preds, min_duration=3, background_id=0)
    assert decoded == [1], f"Expected [1], got {decoded}"
    print("RLE Decode logic: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading...")

    # Instantiate Dataset
    # We force load_cached_data=False initially to test processing logic
    train_ds = GestureDataset(split="train", load_cached_data=False, transform=True)

    assert len(train_ds) == 5, f"Expected 5 samples, got {len(train_ds)}"

    # Get one sample
    sample = train_ds[0]
    skel = sample["skeleton"]
    audio = sample["audio"]
    labels = sample["labels"]

    # Check shapes
    # Skeleton: (T, 60)
    assert (
        skel.ndim == 2 and skel.shape[1] == 60
    ), f"Skeleton shape mismatch: {skel.shape}"
    # Audio: (T, 13)
    assert (
        audio.ndim == 2 and audio.shape[1] == 13
    ), f"Audio shape mismatch: {audio.shape}"
    # Labels: (T,)
    assert labels.ndim == 1, f"Labels shape mismatch: {labels.shape}"
    assert (
        skel.shape[0] == audio.shape[0] == labels.shape[0]
    ), "Temporal dimension mismatch across modalities"

    print(f"Sample loaded successfully. Time steps: {skel.shape[0]}")

    # Test DataLoader Collate
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    batch = next(iter(train_loader))
    b_skel = batch["skeleton"]
    b_lens = batch["lengths"]

    assert b_skel.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert b_skel.shape[1] == b_lens.max(), "Padding length mismatch"
    print("DataLoader batch generation: OK")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Verifying DAGI-Net Model...")

    device = torch.device(Config.DEVICE)
    model = DAGINet().to(device)

    # Move batch to device
    b_skel = b_skel.to(device)
    b_audio = batch["audio"].to(device)
    b_lens = b_lens.to(device)

    # Forward Pass
    logits = model(b_skel, b_audio, b_lens)

    # Check Output: (Batch, Time, NumClasses)
    expected_shape = (Config.BATCH_SIZE, b_skel.shape[1], Config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Logits shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print("Model forward pass: OK")

    # -------------------------------------------------------------------------
    # 5. Verify Training Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Pipeline (Short Run)...")

    # This function handles dataset creation, training loop, validation, and saving
    # We rely on the patched Config to use the subset data and reduced epochs
    run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        load_cached_data=True,  # Use cache generated in step 3
    )

    # Verify checkpoint creation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Training successful. Model saved at: {Config.BEST_MODEL_PATH}")
    else:
        raise AssertionError("Training finished but best_model.pth was not found.")

    print("\n>>> Demonstration Complete.")


if __name__ == "__main__":
    main()
