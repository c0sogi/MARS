import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    rle_decode,
    compute_levenshtein_ratio,
)
from library.data_loader import (
    get_dataloaders,
    GestureDataset,
    collate_fn,
    compute_global_stats,
)
from library.model import MPWINet
from library.engine import train_model, get_loss_function
from library.inference import generate_predictions


def run_demo():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration & Setup
    # Override Config paths and parameters for a fast demo execution
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(DEMO_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "submission"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "metadata"), exist_ok=True)

    # Patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")
    Config.STATS_PATH = os.path.join(DEMO_DIR, "stats.npz")

    # Reduce compute load for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Set Seed
    set_seed(Config.SEED)
    print("Configuration updated for demo execution.")

    # 2. Create Subset Metadata (for speed)
    print("\n[Step 1] Creating data subsets...")

    def create_subset(src_path, dst_name, n=10):
        df = pd.read_csv(src_path)
        # Take top n samples
        subset = df.head(n).copy()
        dst_path = os.path.join(DEMO_DIR, "metadata", dst_name)
        subset.to_csv(dst_path, index=False)
        return dst_path

    # Update Config paths to point to subsets
    Config.TRAIN_METADATA_PATH = create_subset("./metadata/train.csv", "train.csv", n=6)
    Config.VAL_METADATA_PATH = create_subset("./metadata/val.csv", "val.csv", n=4)
    Config.TEST_METADATA_PATH = create_subset("./metadata/test.csv", "test.csv", n=4)
    print("Subset metadata created.")

    # 3. Verify Utility Logic
    print("\n[Step 2] Verifying utility functions...")
    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    assert (
        levenshtein_distance(seq1, seq2) == 0
    ), "Levenshtein distance should be 0 for identical sequences"

    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]  # Substitution
    assert (
        levenshtein_distance(seq1, seq2) == 1
    ), "Levenshtein distance should be 1 for one substitution"

    seq1 = [1, 2]
    seq2 = [1, 2, 3]  # Insertion
    assert (
        levenshtein_distance(seq1, seq2) == 1
    ), "Levenshtein distance should be 1 for insertion"

    # Test RLE Decode
    # Background is 0. Min segment length is usually 5 in config, let's assume input satisfies it.
    # We'll mock a sequence: 1 1 1 1 1 0 0 2 2 2 2 2
    # Assuming Config.MIN_SEGMENT_LENGTH is 5
    raw_preds = [1] * 5 + [0] * 5 + [2] * 5
    decoded = rle_decode(np.array(raw_preds), background_label=0, min_len=5)
    assert decoded == [1, 2], f"RLE Decode failed. Expected [1, 2], got {decoded}"
    print("Utility functions verified.")

    # 4. Data Loading & Statistics
    print("\n[Step 3] Initializing Dataloaders and Computing Stats...")
    # This triggers compute_global_stats on the subset
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Stats file creation
    assert os.path.exists(Config.STATS_PATH), "Stats file was not created."

    # Verify Batch Shapes
    batch = next(iter(train_loader))
    skeleton = batch["skeleton"]
    audio = batch["audio"]
    mask = batch["mask"]
    labels = batch["frame_labels"]

    print(
        f"Batch Shapes -> Skeleton: {skeleton.shape}, Audio: {audio.shape}, Mask: {mask.shape}"
    )

    # Skeleton: (B, T, Joints, Channels) -> (B, T, 20, 3)
    assert (
        skeleton.dim() == 4 and skeleton.shape[2] == 20 and skeleton.shape[3] == 3
    ), "Incorrect skeleton shape"
    # Audio: (B, T, MFCC) -> (B, T, 13)
    assert audio.dim() == 3 and audio.shape[2] == 13, "Incorrect audio shape"
    # Mask: (B, T)
    assert mask.dim() == 2, "Incorrect mask shape"

    print("Data loading verified.")

    # 5. Model Initialization & Forward Pass
    print("\n[Step 4] Model Initialization & Forward Pass...")
    device = Config.DEVICE
    model = MPWINet().to(device)

    # Move batch to device
    skel_dev = skeleton.to(device)
    audio_dev = audio.to(device)
    mask_dev = mask.to(device)
    lengths = batch[
        "lengths"
    ]  # Keep on CPU for packing logic inside model if needed, or model handles it

    # Forward
    logits = model(skel_dev, audio_dev, lengths, mask_dev)
    print(f"Logits Shape: {logits.shape}")

    # Expected: (B, T, NumClasses)
    assert logits.shape[0] == skeleton.shape[0], "Batch dimension mismatch"
    assert logits.shape[1] == skeleton.shape[1], "Time dimension mismatch"
    assert (
        logits.shape[2] == Config.NUM_CLASSES
    ), f"Class dimension mismatch. Expected {Config.NUM_CLASSES}, got {logits.shape[2]}"
    print("Model forward pass verified.")

    # 6. Training Loop Execution
    print("\n[Step 5] Running Training Loop...")
    # train_model runs epochs and validation
    best_model_path = train_model(train_loader, val_loader)

    assert os.path.exists(
        best_model_path
    ), "Best model checkpoint not found after training."
    print(f"Training completed. Model saved at {best_model_path}")

    # 7. Inference & Submission
    print("\n[Step 6] Running Inference...")
    generate_predictions(
        model_path=best_model_path,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated."

    # Check content of submission
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"Submission file has {len(lines)} lines.")
        # We expect 1 line per test sample (4 samples in subset)
        assert len(lines) == 4, f"Expected 4 predictions, got {len(lines)}"

    print("Inference pipeline verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
