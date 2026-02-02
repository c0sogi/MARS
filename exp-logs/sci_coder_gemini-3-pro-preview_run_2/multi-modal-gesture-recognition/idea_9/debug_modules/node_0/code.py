import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import GestureDataset, collate_fn
from library.model import MDCRCN
from library.loss import MaskedWeightedCrossEntropy, TMSELoss
from library.trainer import Trainer
from library.inference import Predictor


def main():
    print("=== MD-CRCN Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Override Config for isolation and speed
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Re-create demo directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update file paths
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Reduce hyperparameters for quick execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 1

    # Set seed and device
    set_seed(42)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load a tiny subset of training data (8 samples)
    # load_cached_data=False forces processing from scratch to test extraction logic
    train_dataset = GestureDataset(split="train", load_cached_data=False, limit=8)
    print(f"    Loaded {len(train_dataset)} training samples.")

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn
    )

    # Fetch one batch
    features, labels, mask, sample_ids = next(iter(loader))

    print(f"    Features Shape: {features.shape}")  # Expected: (B, T, 85)
    print(f"    Labels Shape:   {labels.shape}")  # Expected: (B, T)
    print(f"    Mask Shape:     {mask.shape}")  # Expected: (B, T)

    # Assertions
    assert features.dim() == 3, "Features must be 3D (Batch, Time, Feats)"
    assert (
        features.size(2) == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {features.size(2)}"
    assert labels.dim() == 2, "Labels must be 2D (Batch, Time)"
    assert mask.dim() == 2, "Mask must be 2D (Batch, Time)"
    assert len(sample_ids) == Config.BATCH_SIZE, "Sample ID count mismatch"
    print("    Data Loader checks passed.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    model = MDCRCN().to(device)
    features = features.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    # Forward Pass
    outputs = model(features, mask)

    # Check outputs
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    logits = outputs["stage3"]
    print(f"    Logits Shape: {logits.shape}")  # Expected: (B, T, 21)

    assert logits.size(0) == Config.BATCH_SIZE
    assert logits.size(2) == Config.NUM_CLASSES
    assert logits.size(1) == features.size(
        1
    ), "Temporal dimension mismatch between input and output"

    # Loss Calculation
    ce_loss_fn = MaskedWeightedCrossEntropy().to(device)
    tmse_loss_fn = TMSELoss().to(device)

    loss_ce = ce_loss_fn(logits, labels, mask)
    probs = torch.softmax(logits, dim=2)
    loss_tmse = tmse_loss_fn(probs, mask)

    print(f"    CE Loss: {loss_ce.item():.4f}")
    print(f"    TMSE Loss: {loss_tmse.item():.4f}")

    assert not torch.isnan(loss_ce), "CE Loss is NaN"
    assert not torch.isnan(loss_tmse), "TMSE Loss is NaN"
    print("    Model and Loss checks passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[4] Simulating Training Loop (1 Epoch)...")

    # Initialize Trainer with a small data limit
    trainer = Trainer(load_cached_data=False, limit=16)

    # Run training
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Check if model checkpoint exists
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"    Checkpoint saved at: {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Best model checkpoint was not created.")
    print("    Training simulation passed.")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference...")

    predictor = Predictor(model_path=Config.BEST_MODEL_PATH)

    # Run inference on a tiny subset of test data
    predictor.run_inference(load_cached_data=False, limit=5)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Submission saved at: {Config.SUBMISSION_PATH}")

        # Validate content format
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"    Generated {len(lines)} predictions.")
            if lines:
                sample_line = lines[0].strip()
                print(f"    Sample Output: {sample_line}")
                parts = sample_line.split(",")
                # Check that first part looks like a Session ID (e.g., Sample00300)
                assert "Sample" in parts[0], "Invalid SessionID format in submission"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("    Inference checks passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
