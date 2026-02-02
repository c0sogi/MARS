import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    run_length_encoding,
    TruncatedMSELoss,
)
from library.data_loader import get_dataloaders
from library.model import RHCKN
from library.train import (
    get_loss_criterion,
    compute_loss,
    train_one_epoch,
    validate,
    generate_submission,
)


def demo_utils():
    """Validates utility functions logic."""
    print("\n=== Testing Utilities ===")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1 for deletion, got {dist_diff}"
    print("Levenshtein distance check passed.")

    # 2. Run Length Encoding
    # Config default min_duration is 5
    # Sequence: 5x '1', 4x '2', 6x '3', 5x '0' (Background)
    raw_preds = [1] * 5 + [2] * 4 + [3] * 6 + [0] * 5
    encoded = run_length_encoding(raw_preds, min_duration=5, background_class=0)

    # Expect: [1, 3] (2 is too short, 0 is background)
    expected = [1, 3]
    assert encoded == expected, f"RLE failed. Expected {expected}, got {encoded}"
    print("Run-Length Encoding check passed.")


def demo_pipeline():
    """Demonstrates the full training and inference pipeline."""
    print("\n=== Starting Pipeline Demo ===")

    # 1. Setup Configuration
    # We use a specific demo directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Enable Debug Mode for speed (Small subset, fewer epochs)
    Config.set_debug(True)
    Config.setup()

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n--- Loading Data (Debug Mode) ---")
    # load_cached_data=False forces processing to verify data loader logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Batch Structure
    batch = next(iter(train_loader))
    features = batch["feature"]
    labels = batch["label"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Feature shape: {features.shape}")  # (Batch, Time, Features)
    print(f"Label shape: {labels.shape}")  # (Batch, Time)

    assert features.dim() == 3, "Features should be 3D tensor (B, T, F)"
    assert labels.dim() == 2, "Labels should be 2D tensor (B, T)"
    assert (
        features.shape[2] == Config.TOTAL_INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.TOTAL_INPUT_DIM}, got {features.shape[2]}"

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = RHCKN().to(device)

    # Forward Pass Check
    features = features.to(device)
    outputs = model(features)

    assert (
        "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    ), "Model output missing stages."

    s3_out = outputs["stage3"]
    assert s3_out.shape == (
        features.shape[0],
        features.shape[1],
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Got {s3_out.shape}"
    print("Forward pass successful.")

    # 4. Loss Computation
    print("\n--- Computing Loss ---")
    ce_criterion, smooth_criterion = get_loss_criterion(device)
    labels = labels.to(device)

    loss = compute_loss(outputs, labels, ce_criterion, smooth_criterion)
    print(f"Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Training Loop (Single Epoch)
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # We run just one epoch for demonstration
    train_loss = train_one_epoch(
        model, train_loader, optimizer, ce_criterion, smooth_criterion, device
    )
    print(f"Train Loss: {train_loss:.4f}")

    # 6. Validation
    print("\n--- Running Validation ---")
    val_score = validate(model, val_loader, device)
    print(f"Validation Levenshtein Score: {val_score:.4f}")

    # 7. Inference / Submission
    print("\n--- Generating Submission ---")
    # Save dummy model state to simulate loading best model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Read first few lines of submission to verify format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"Submission lines generated: {len(lines)}")
        if len(lines) > 0:
            print(f"Sample line: {lines[0].strip()}")
            parts = lines[0].strip().split(",")
            # Format: SessionID, label1, label2...
            # SessionID should be string, labels integers
            assert len(parts) >= 1, "Invalid submission line format"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure clean state
    if os.path.exists("./working/demo_execution"):
        shutil.rmtree("./working/demo_execution")

    demo_utils()
    demo_pipeline()
