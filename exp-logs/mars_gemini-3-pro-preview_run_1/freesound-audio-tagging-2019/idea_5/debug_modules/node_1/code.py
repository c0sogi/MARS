import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_lwlrap, calculate_per_class_lwlrap
from library.dataset import get_dataloaders, AudioDataset
from library.model import AudioEfficientNet
from library.engine import (
    train_one_epoch,
    validate,
    inference,
    generate_submission,
    get_or_compute_teacher_predictions,
)


def run_demonstration():
    print("=== Starting Audio Tagging Pipeline Demonstration ===")

    # 1. Setup & Configuration Override
    # We modify the Config class state directly to optimize for a quick demo run.
    print("\n[1] Configuring environment...")
    set_seed(42)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use main thread to avoid overhead in small demo
    Config.PROJECT_NAME = "demo_project"
    Config.OUTPUT_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Ensure directories exist
    Config.setup()

    # 2. Verify Metric Logic
    print("\n[2] Verifying Metric Logic (LWLRAP)...")
    # Case 1: Perfect prediction
    truth = np.array([[1, 0, 1], [0, 1, 0]])
    scores_perfect = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]])
    lwlrap_perfect = calculate_lwlrap(truth, scores_perfect)
    print(f"   Perfect Score LWLRAP: {lwlrap_perfect:.4f}")
    assert np.isclose(
        lwlrap_perfect, 1.0
    ), "Metric calculation failed for perfect predictions."

    # Case 2: Worst prediction (inverse)
    scores_worst = np.array([[0.1, 0.9, 0.2], [0.9, 0.1, 0.8]])
    lwlrap_worst = calculate_lwlrap(truth, scores_worst)
    print(f"   Worst Score LWLRAP: {lwlrap_worst:.4f}")
    # Calculation:
    # Item 1 (True: 0, 2): Ranks [3, 1, 2].
    #   Rank 2 (Class 2): 1/2 prec. Rank 3 (Class 0): 2/3 prec. Avg: (0.5+0.66)/2 = 0.58
    # Item 2 (True: 1): Ranks [1, 3, 2].
    #   Rank 3 (Class 1): 1/3 prec. Avg: 0.33
    # Mean: ~0.45. Just ensuring it runs and is low.
    assert lwlrap_worst < 1.0, "Metric calculation failed for bad predictions."
    print("   Metric verification passed.")

    # 3. Data Loading & Dataset
    print("\n[3] Loading Data and Creating DataLoaders...")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    print(f"   Original Train size: {len(train_df)}")

    # Subset for demonstration (First 12 samples to allow batch_size=4 to drop_last=True if needed)
    train_subset = train_df.head(12).copy()
    val_subset = val_df.head(8).copy()
    test_subset = test_df.head(8).copy()

    print(f"   Subset Train size: {len(train_subset)}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_subset, val_subset, test_subset
    )

    # Fetch one batch to verify shapes
    print("   Fetching one batch from Train Loader...")
    specs, targets, fnames = next(iter(train_loader))

    # Expected Spectrogram Shape: (Batch, 1, n_mels, time)
    # Time dimension depends on Config.DURATION (5s) * Config.SR (32000) / Config.HOP_LENGTH (320)
    # 160000 / 320 = 500 time steps approx.
    print(f"   Spectrogram Shape: {specs.shape}")
    print(f"   Targets Shape: {targets.shape}")

    assert specs.shape[0] == Config.BATCH_SIZE
    assert specs.shape[1] == 1
    assert specs.shape[2] == Config.N_MELS
    assert targets.shape[1] == Config.NUM_CLASSES
    print("   Data shapes verified.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    device = Config.DEVICE
    print(f"   Device: {device}")

    model = AudioEfficientNet(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Forward pass verification
    print("   Running forward pass on sample batch...")
    specs = specs.to(device)
    with torch.no_grad():
        outputs = model(specs)

    print(f"   Output Logits Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("   Model forward pass successful.")

    # 5. Training Loop Simulation
    print("\n[5] Simulating Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Train
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=1
    )
    print(f"   Train Loss: {avg_train_loss:.6f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"

    # Validate
    avg_val_loss, val_lwlrap = validate(model, val_loader, device)
    print(f"   Val Loss: {avg_val_loss:.6f}")
    print(f"   Val LWLRAP: {val_lwlrap:.6f}")
    assert 0 <= val_lwlrap <= 1.0, "Validation metric out of range"

    # 6. Teacher Predictions (Distillation Helper)
    print("\n[6] Testing Teacher Prediction Generation (Soft Labels)...")
    # We use the val_loader as a proxy for 'noisy' data loader for this demo
    cache_path = os.path.join(Config.OUTPUT_DIR, "demo_teacher_preds.npy")

    teacher_preds = get_or_compute_teacher_predictions(
        model, val_loader, device, cache_path, load_cached_data=False
    )

    print(f"   Generated predictions for {len(teacher_preds)} files.")
    assert len(teacher_preds) == len(val_subset)
    # Verify shape of one prediction
    first_key = list(teacher_preds.keys())[0]
    assert teacher_preds[first_key].shape == (Config.NUM_CLASSES,)

    # Verify file exists
    assert os.path.exists(cache_path), "Cache file was not saved."
    print("   Teacher predictions computed and cached successfully.")

    # 7. Inference & Submission
    print("\n[7] Generating Submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission shape: {sub_df.shape}")
    # Shape should be (N_test, 1 + N_classes) -> 1 for fname
    expected_cols = Config.NUM_CLASSES + 1
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"
    assert sub_df.shape[0] == len(
        test_subset
    ), f"Expected {len(test_subset)} rows, got {sub_df.shape[0]}"

    print("   Submission generated successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
