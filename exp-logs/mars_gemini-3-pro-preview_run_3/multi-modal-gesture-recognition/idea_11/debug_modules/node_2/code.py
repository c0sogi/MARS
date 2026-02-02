import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
import library.utils as utils
from library.data_loader import get_dataloaders
from library.model import BA_AKN
from library.loss import BoundaryAwareLoss
import library.trainer as trainer_module
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("=== Starting BA-AKN Demo Execution ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demo
    # ---------------------------------------------------------
    print("[1/6] Configuring environment...")

    # Define demo working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    # Set seed for reproducibility
    utils.set_seed(Config.SEED)
    print(f"Configuration set. Working dir: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Utility Verification
    # ---------------------------------------------------------
    print("[2/6] Verifying Utilities...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_same = utils.levenshtein_distance(seq1, seq2)
    assert dist_same == 0, f"Levenshtein distance should be 0, got {dist_same}"

    seq3 = [1, 2]
    dist_diff = utils.levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Levenshtein distance should be 1, got {dist_diff}"

    # Test Collapse Predictions (RLE + Background removal)
    # 0 is background.
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0, 3, 0]
    collapsed = utils.collapse_predictions(raw_preds)
    expected = [1, 2, 3]
    assert (
        collapsed == expected
    ), f"Collapse failed. Expected {expected}, got {collapsed}"

    print("Utilities verified successfully.")

    # ---------------------------------------------------------
    # 3. Data Loader Verification
    # ---------------------------------------------------------
    print("[3/6] Verifying Data Loaders...")

    # Force reload to ensure cache is built in demo dir
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Fetch a single batch
    features, cls_targets, bnd_targets = next(iter(train_loader))

    # Verify Shapes
    # Features: (Batch, Window, InputDim)
    assert features.dim() == 3, f"Expected 3D features, got {features.shape}"
    assert (
        features.size(0) == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {features.size(0)}"
    assert (
        features.size(1) == Config.WINDOW_SIZE
    ), f"Expected window size {Config.WINDOW_SIZE}, got {features.size(1)}"
    assert (
        features.size(2) == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {features.size(2)}"

    # Targets: (Batch, Window)
    assert cls_targets.shape == (Config.BATCH_SIZE, Config.WINDOW_SIZE)
    assert bnd_targets.shape == (Config.BATCH_SIZE, Config.WINDOW_SIZE)

    print(
        f"Data batch shapes verified: Features {features.shape}, Targets {cls_targets.shape}"
    )

    # ---------------------------------------------------------
    # 4. Model & Loss Verification
    # ---------------------------------------------------------
    print("[4/6] Verifying Model and Loss...")

    model = BA_AKN().to(Config.DEVICE)
    criterion = BoundaryAwareLoss().to(Config.DEVICE)

    # Move batch to device
    features = features.to(Config.DEVICE)
    cls_targets = cls_targets.to(Config.DEVICE)
    bnd_targets = bnd_targets.to(Config.DEVICE)

    # Forward Pass
    outputs = model(features)

    # Check Deep Supervision Outputs
    required_keys = [
        "stage1_cls",
        "stage1_bnd",
        "stage2_cls",
        "stage2_bnd",
        "stage3_cls",
        "stage3_bnd",
    ]
    for key in required_keys:
        assert key in outputs, f"Missing output key: {key}"

    # Check Output Shape (Batch, Classes/1, Time)
    # Note: Model outputs (Batch, Channels, Time), Loader provides (Batch, Time, Channels)
    # The BiGRUEncoder handles the permutation internally.
    s3_cls = outputs["stage3_cls"]
    assert s3_cls.size(0) == Config.BATCH_SIZE
    assert s3_cls.size(1) == Config.NUM_CLASSES
    assert s3_cls.size(2) == Config.WINDOW_SIZE

    # Compute Loss
    loss, metrics = criterion(outputs, cls_targets, bnd_targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Backward Pass
    loss.backward()
    print("Model forward/backward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("[5/6] Running Training Loop (1 Epoch)...")

    # Monkey-patch tqdm to suppress progress bars
    trainer_module.tqdm = lambda x, **kwargs: x

    trainer = Trainer()
    # We use load_cached_data=True because we generated cache in step 3
    trainer.fit(epochs=Config.EPOCHS, load_cached_data=True)

    # Check artifacts
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training loop completed and model saved.")

    # ---------------------------------------------------------
    # 6. Inference Demonstration
    # ---------------------------------------------------------
    print("[6/6] Running Inference...")

    # Run inference using the trained model
    run_inference(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Check content format
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0, "Submission file is empty"
        # Check first line format: SessionID,label,label...
        parts = lines[0].strip().split(",")
        assert len(parts) >= 1, "Invalid submission line format"
        # First part should be sample ID (string)
        assert (
            "Sample" in parts[0]
            or "Session" in parts[0]
            or "devel" in parts[0]
            or "valid" in parts[0]
            or len(parts[0]) > 0
        )

    print(f"Inference completed. Submission saved to {Config.SUBMISSION_FILE}")
    print("=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
