import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    rle_decode,
    compute_levenshtein_score,
)
from library.data_loader import GestureDataset, get_dataloaders, collate_fn
from library.model import MSC_IIN
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration Overrides for Demo
    print(">>> 1. Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.SEED = 42
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.MAX_SAMPLES = 20  # Limit to 20 samples for speed
    Config.DEBUG = True

    # Redirect working directory to avoid messing with real training artifacts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed()
    print("Configuration updated.")

    # 2. Test Utilities
    print("\n>>> 2. Testing Utility Functions...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for equal seqs should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Levenshtein distance should be 1, got {dist_diff}"
    print("Levenshtein distance logic verified.")

    # Test RLE Decode
    # 0 is background. Min length is usually 5 in Config, let's assume default.
    # Sequence: 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2 (2 is too short)
    raw_preds = [0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2]
    decoded = rle_decode(raw_preds, background_id=0, min_length=5)
    # Should keep 1 (length 5), ignore 2 (length 3)
    assert decoded == [1], f"RLE Decode failed. Expected [1], got {decoded}"
    print("RLE Decode logic verified.")

    # 3. Test Data Loading
    print("\n>>> 3. Testing Data Loading...")

    # Instantiate Dataset
    train_ds = GestureDataset(split="train", max_samples=Config.MAX_SAMPLES)
    print(f"Train dataset size: {len(train_ds)}")

    if len(train_ds) > 0:
        sample = train_ds[0]
        # Verify keys
        required_keys = ["skeleton", "audio", "labels", "length", "id"]
        for k in required_keys:
            assert k in sample, f"Sample missing key: {k}"

        # Verify shapes
        # Skeleton: (T, 60)
        assert sample["skeleton"].shape[1] == Config.SKELETON_INPUT_CHANNELS
        # Audio: (T, 13)
        assert sample["audio"].shape[1] == Config.N_MFCC
        # Labels: (T,)
        assert sample["labels"].shape[0] == sample["skeleton"].shape[0]

        print(
            f"Sample 0 shapes verified: Skel {sample['skeleton'].shape}, Audio {sample['audio'].shape}"
        )
    else:
        print(
            "Warning: Dataset is empty (check metadata). Skipping sample verification."
        )

    # Test DataLoader & Collate
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn
    )
    batch = next(iter(loader))

    if batch is not None:
        # Check batch shapes
        # (B, T_max, C)
        B = len(batch["ids"])
        T_max = batch["skeleton"].shape[1]
        assert batch["skeleton"].shape == (B, T_max, Config.SKELETON_INPUT_CHANNELS)
        assert batch["audio"].shape == (B, T_max, Config.N_MFCC)
        assert batch["labels"].shape == (B, T_max)
        assert batch["lengths"].shape[0] == B
        print("Batch collation verified.")

    # 4. Test Model Forward Pass
    print("\n>>> 4. Testing Model Architecture...")
    device = Config.get_device()
    model = MSC_IIN().to(device)

    if batch is not None:
        skel = batch["skeleton"].to(device)
        aud = batch["audio"].to(device)
        lens = batch["lengths"].to(device)

        logits = model(skel, aud, lens)

        # Expected output: (B, T, NumClasses)
        assert logits.shape == (
            B,
            T_max,
            Config.NUM_CLASSES,
        ), f"Model output shape mismatch. Expected {(B, T_max, Config.NUM_CLASSES)}, got {logits.shape}"

        print("Model forward pass verified.")

    # 5. Test Trainer (Full Loop)
    print("\n>>> 5. Testing Trainer Execution (Train/Val/Test)...")

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    # This will run for Config.NUM_EPOCHS (2) on the small dataset
    trainer.run()

    # Verify Checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        print("Checkpoint file created successfully.")
    else:
        print(
            "Note: No best model checkpoint found (might be due to empty val set or high error)."
        )

    # Generate Submission
    trainer.generate_submission()

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        # Check content
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                print(
                    f"Submission has {len(lines)} lines. First line: {lines[0].strip()}"
                )
            else:
                print("Warning: Submission file is empty.")
    else:
        raise AssertionError("Submission file was not created.")

    print("\n>>> Demonstration completed successfully.")


if __name__ == "__main__":
    main()
