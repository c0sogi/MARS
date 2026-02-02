import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")
import tqdm


def nop(it, *a, **k):
    return it


tqdm.tqdm = nop

# Import library components
from library.config import Config
from library.utils import set_seed, compute_levenshtein_distance
from library.data_loader import GestureDataset
from library.model import ACGRNet
from library.train import Trainer, robust_collate_fn, load_dense_labels
from library.inference import generate_predictions


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Setting up demonstration configuration...")

    # Override Config for speed and isolation
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache_demo")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORK_DIR  # Save submission in demo dir
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Lightweight Model & Training params
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_DIM = 64
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.1

    # Re-run setup to create new directories
    Config.setup()
    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Loading & Validation
    # ==========================================
    print("Validating data loading pipeline...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Use subsets for speed
    subset_size = 20
    train_subset = train_df.head(subset_size)
    val_subset = val_df.head(subset_size)

    # Save subsets to temporary CSVs to trick the Dataset class into loading fewer files
    # (The Dataset class reads from Config paths, so we temporarily mock them or just rely on max_samples)
    # Since Dataset takes max_samples, we use that feature.

    # Pre-compute dense labels for the subset (required for training)
    # We combine train and val for cache population
    combined_subset = pd.concat([train_subset, val_subset], ignore_index=True)
    cache_path = os.path.join(Config.WORK_DIR, "dense_labels_demo.npy")
    load_dense_labels(combined_subset, cache_path, load_cached_data=False)

    # Instantiate Datasets
    train_dataset = GestureDataset(
        split="train", load_cached_data=False, max_samples=subset_size
    )
    val_dataset = GestureDataset(
        split="val", load_cached_data=False, max_samples=subset_size
    )

    assert len(train_dataset) <= subset_size, "Train dataset size mismatch"

    # Instantiate Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=robust_collate_fn,
        num_workers=0,  # Avoid multiprocessing overhead for demo
    )

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {
        "skeleton",
        "audio",
        "dense_labels",
        "seq_labels",
        "lengths",
        "mask",
        "sample_ids",
    }
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch: {expected_keys - set(batch.keys())}"

    # Verify Shapes
    # skeleton: (B, T, 60)
    # audio: (B, T, 13)
    # dense_labels: (B, T)
    B, T, D_skel = batch["skeleton"].shape
    _, _, D_audio = batch["audio"].shape

    assert B == Config.BATCH_SIZE or B == len(
        train_dataset
    ), f"Batch size mismatch: {B}"
    assert D_skel == Config.INPUT_DIM_SKELETON, f"Skeleton dim mismatch: {D_skel}"
    assert D_audio == Config.INPUT_DIM_AUDIO, f"Audio dim mismatch: {D_audio}"
    assert batch["dense_labels"].shape == (B, T), "Dense labels shape mismatch"
    assert batch["mask"].shape == (B, T), "Mask shape mismatch"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Model Verification
    # ==========================================
    print("Verifying model architecture...")

    model = ACGRNet().to(Config.DEVICE)

    # Forward pass
    skeleton = batch["skeleton"].to(Config.DEVICE)
    audio = batch["audio"].to(Config.DEVICE)
    mask = batch["mask"].to(Config.DEVICE)

    logits = model(skeleton, audio, mask)

    # Output shape should be (B, T, NumClasses)
    assert logits.shape == (
        B,
        T,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {logits.shape}"

    print("Model forward pass verified.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("Running training loop demonstration...")

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=robust_collate_fn,
        num_workers=0,
    )

    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Verify Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training loop completed and checkpoint saved.")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("Running inference demonstration...")

    # We use the generate_predictions function from library.inference
    # It internally creates a test dataset. We limit samples for speed.
    # Note: generate_predictions uses Config.SUBMISSION_PATH which we overrode.

    # We need to ensure the test dataset loads correctly.
    # The library function creates a GestureDataset(split='test').
    # We will call it directly.

    generate_predictions(load_cached_data=False, max_samples=5)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Verify content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0, "Submission file is empty."
        # Check first line format: SessionID,Labels
        parts = lines[0].strip().split(",")
        assert len(parts) >= 1, "Invalid submission line format."
        assert parts[0].startswith("Sample") or parts[0].startswith(
            "Session"
        ), "Invalid ID in submission."

    print("Inference verified.")

    # ==========================================
    # 6. Metric Logic Verification
    # ==========================================
    print("Verifying metric logic...")

    # Case 1: Identical
    s1 = [1, 2, 3]
    s2 = [1, 2, 3]
    # Dist should be 0, normalized 0
    d = compute_levenshtein_distance([s1], [s2])
    assert d == 0.0, f"Metric failed for identical sequences. Got {d}"

    # Case 2: One insertion
    s1 = [1, 2]
    s2 = [1, 2, 3]
    # Dist is 1. Total length is 3. Result 1/3.
    d = compute_levenshtein_distance([s1], [s2])
    expected = 1.0 / 3.0
    assert (
        abs(d - expected) < 1e-6
    ), f"Metric failed for insertion. Got {d}, expected {expected}"

    # Case 3: Empty prediction
    s1 = []
    s2 = [1, 2]
    # Dist is 2. Total len 2. Result 1.0.
    d = compute_levenshtein_distance([s1], [s2])
    assert abs(d - 1.0) < 1e-6, f"Metric failed for empty pred. Got {d}"

    print("Metric logic verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
