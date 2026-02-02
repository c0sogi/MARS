import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. Configuration Override
# -----------------------------------------------------------------------------
# We must import library.config and modify it BEFORE importing other modules
# so that they pick up the modified values.
import library.config

# Define working paths in the writable ./working directory
DEMO_DIR = "./working/demo_run"
CACHE_DIR = os.path.join(DEMO_DIR, "cache")
SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "demo_submission.csv")

# Clean up previous run to ensure a fresh start
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Override Config for Speed and Demo purposes
library.config.DEBUG_SUBSET_SIZE = 12  # Process only a small subset of data
library.config.BATCH_SIZE = 4  # Small batch size
library.config.NUM_EPOCHS = 1  # Only 1 epoch for demonstration
library.config.CACHE_DIR = CACHE_DIR
library.config.SUBMISSION_DIR = SUBMISSION_DIR
library.config.SUBMISSION_FILE = SUBMISSION_FILE

# Reduce model architecture size for faster execution on CPU/GPU
library.config.HIDDEN_SIZE = 64
library.config.NUM_LSTM_LAYERS = 1
library.config.NUM_TCN_LAYERS = 2

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.utils import set_seed, compute_levenshtein
from library.data import get_loaders
from library.model import DSG_CRCN
from library.loss import ActionSegmentationLoss
from library.engine import Trainer


def run_demo():
    print("=== Starting DSG-CRCN Demo ===")

    # Set seed for reproducibility
    set_seed(42)
    device = library.config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 3. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing Data Loaders...")
    # load_cached_data=False forces the dataset to process raw files (verifying data pipeline)
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch a single batch to verify structure
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check data paths or subset size.")

    features = batch["features"]
    cls_labels = batch["cls_labels"]
    mask = batch["mask"]

    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Features shape: {features.shape}")  # Expected: (B, T, InputDim)
    print(f"  Labels shape: {cls_labels.shape}")  # Expected: (B, T)

    # Assertions
    assert features.ndim == 3, f"Expected features dim 3, got {features.ndim}"
    assert cls_labels.ndim == 2, f"Expected labels dim 2, got {cls_labels.ndim}"
    assert mask.ndim == 2, f"Expected mask dim 2, got {mask.ndim}"
    print("  Data loading verified.")

    # -------------------------------------------------------------------------
    # 4. Model Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Forward Pass...")
    model = DSG_CRCN().to(device)

    # Move data to device
    f_dev = features.to(device)
    m_dev = mask.to(device)

    # Forward pass
    outputs = model(f_dev, m_dev)

    print(f"  Output keys: {list(outputs.keys())}")

    # Check outputs
    assert "final_cls" in outputs, "Missing 'final_cls' in model output"
    assert "final_bnd" in outputs, "Missing 'final_bnd' in model output"

    final_cls = outputs["final_cls"]
    print(f"  Final Classification Shape: {final_cls.shape}")

    # Shape assertions: (B, T, NumClasses)
    assert final_cls.shape[0] == features.shape[0], "Batch size mismatch"
    assert final_cls.shape[1] == features.shape[1], "Sequence length mismatch"
    assert final_cls.shape[2] == library.config.NUM_CLASSES, "Class dimension mismatch"
    print("  Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Calculation Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Loss Calculation...")
    criterion = ActionSegmentationLoss()

    targets = {
        "cls_labels": cls_labels.to(device),
        "bnd_labels": batch["bnd_labels"].to(device),
        "mask": m_dev,
    }

    loss, metrics = criterion(outputs, targets)
    print(f"  Loss Value: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("  Loss calculation verified.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    trainer = Trainer()

    # Verify that the trainer initialized the model with our overridden config
    # The hidden size of the LSTM in stage 1 should match our override (64)
    model_hidden = trainer.model.stage1.lstm.hidden_size
    assert (
        model_hidden == 64
    ), f"Trainer model hidden size {model_hidden} != 64. Config override failed."

    # Run training
    trainer.fit(train_loader, val_loader, epochs=1)

    # Check if best model checkpoint was created
    # Note: It is created if validation loss < infinity (which it should be)
    if os.path.exists(trainer.best_model_path):
        print(f"  Checkpoint successfully created at: {trainer.best_model_path}")
    else:
        # If for some reason validation failed completely, this might not exist
        print(
            "  Warning: Checkpoint not found (possibly due to empty val set or error)."
        )

    print("  Training loop execution verified.")

    # -------------------------------------------------------------------------
    # 7. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Inference and Generating Submission...")
    trainer.predict(test_loader, output_path=SUBMISSION_FILE)

    assert os.path.exists(SUBMISSION_FILE), "Submission file was not created"

    # Validate submission format
    df = pd.read_csv(SUBMISSION_FILE)
    print("  Submission File Head:")
    print(df.head())

    assert "Id" in df.columns, "Submission missing 'Id' column"
    assert "Sequence" in df.columns, "Submission missing 'Sequence' column"
    assert len(df) == len(
        test_loader.dataset
    ), f"Expected {len(test_loader.dataset)} rows, got {len(df)}"
    print("  Inference pipeline verified.")

    # -------------------------------------------------------------------------
    # 8. Metric Utility Verification
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Metric Utility (Levenshtein)...")
    # Example:
    # Pred: [1, 2, 3]
    # Targ: [1, 2, 4]
    # Distance is 1 (Substitution of 3->4)
    # Total length = 3
    # Score = 1 / 3 = 0.333...
    preds = [[1, 2, 3]]
    targs = [[1, 2, 4]]

    score = compute_levenshtein(preds, targs)
    print(f"  Computed Score: {score:.4f}")
    assert (
        abs(score - 0.3333) < 0.01
    ), f"Metric calculation incorrect. Expected ~0.333, got {score}"
    print("  Metric utility verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
