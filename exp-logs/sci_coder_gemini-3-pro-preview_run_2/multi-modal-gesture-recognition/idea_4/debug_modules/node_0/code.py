import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.utils import (
    seed_everything,
    smooth_predictions,
    decode_sequence,
    compute_normalized_levenshtein,
    compute_levenshtein,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import TemporalTransformer
from library.trainer import Trainer


def run_demo():
    # 1. Setup and Reproducibility
    print(">>> Setting up and seeding...")
    seed_everything(42)

    # Define paths
    working_dir = "./working/demo_run"
    cache_dir = os.path.join(working_dir, "cache")
    submission_dir = os.path.join(working_dir, "submission")

    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Demonstrate Utils
    print("\n>>> Testing Utils...")

    # Test smoothing and decoding
    # Sequence: Background(0) -> Class 1 -> Jitter -> Class 1 -> Background -> Class 2 -> Background
    raw_preds = np.array([0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 2, 2, 0])
    # Smooth with window 3: median filtering should remove the single '2' in the middle of 1s
    smoothed = smooth_predictions(raw_preds, window_size=3)

    print(f"Raw predictions: {raw_preds}")
    print(f"Smoothed (w=3):  {smoothed}")

    decoded = decode_sequence(smoothed, background_class_id=0)
    print(f"Decoded sequence: {decoded}")

    # Verify Levenshtein
    target = [1, 2]
    dist = compute_levenshtein(decoded, target)
    print(f"Levenshtein distance to {target}: {dist}")

    # Assertions
    assert isinstance(decoded, list)
    assert isinstance(dist, int)

    # 3. Demonstrate Data Loader
    print("\n>>> Testing Data Loader...")

    # Use a tiny subset of the real metadata
    train_metadata_file = "./metadata/train.csv"
    val_metadata_file = "./metadata/val.csv"
    test_metadata_file = "./metadata/test.csv"

    # Load datasets with max_samples to speed up
    batch_size = 2
    train_dataset = GestureDataset(
        metadata_file=train_metadata_file,
        root_dir="./input",
        cache_dir=cache_dir,
        load_cached_data=False,  # Force processing to test logic
        max_samples=4,
    )

    val_dataset = GestureDataset(
        metadata_file=val_metadata_file,
        root_dir="./input",
        cache_dir=cache_dir,
        load_cached_data=False,
        max_samples=4,
    )

    test_dataset = GestureDataset(
        metadata_file=test_metadata_file,
        root_dir="./input",
        cache_dir=cache_dir,
        load_cached_data=False,
        is_test=True,
        max_samples=4,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    assert len(train_dataset) == 4

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Fetch one batch
    features, labels, lengths, mask = next(iter(train_loader))

    print(f"Batch Features Shape: {features.shape} (Batch, MaxLen, FeatDim)")
    print(f"Batch Labels Shape: {labels.shape} (Batch, MaxLen)")
    print(f"Batch Lengths: {lengths}")
    print(f"Batch Mask Shape: {mask.shape}")

    # Assertions
    # Input dim should be 85 (72 skeleton + 13 audio)
    assert features.shape[2] == 85
    assert features.shape[0] == batch_size
    assert labels.shape[0] == batch_size
    assert mask.shape == labels.shape

    # 4. Demonstrate Model
    print("\n>>> Testing Model...")

    config = {
        "input_dim": 85,
        "num_classes": 21,
        "d_model": 32,  # Small for demo
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "epochs": 1,
        "patience": 1,
        "noise_std": 0.0,
    }

    model = TemporalTransformer(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
    )

    # Forward pass on CPU
    padding_mask = ~mask
    output = model(features, src_key_padding_mask=padding_mask)

    print(f"Model Output Shape: {output.shape} (Batch, MaxLen, NumClasses)")
    assert output.shape == (batch_size, features.shape[1], config["num_classes"])

    # 5. Demonstrate Trainer
    print("\n>>> Testing Trainer...")

    trainer = Trainer(config)

    # Override paths to keep things clean and isolated
    trainer.working_dir = working_dir
    trainer.checkpoint_path = os.path.join(working_dir, "best_model.pth")

    # Re-create loaders for trainer (using the mini datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Fit
    print("Running training loop (1 epoch)...")
    trainer.fit(train_loader, val_loader, epochs=1, patience=1)

    # Predict
    output_csv = os.path.join(submission_dir, "demo_submission.csv")
    print("Running prediction...")
    trainer.predict(test_loader, output_file=output_csv)

    # Verify submission file
    assert os.path.exists(output_csv)
    with open(output_csv, "r") as f:
        lines = f.readlines()
        print(f"Submission file created with {len(lines)} lines.")
        if len(lines) > 0:
            print(f"First line: {lines[0].strip()}")

    # Check if number of lines matches test samples (4)
    assert len(lines) == 4

    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
