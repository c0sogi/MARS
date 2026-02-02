import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
import library.config as config
import library.trainer as trainer_module
from library.data_loader import GestureDataset, collate_fn
from library.model import GSG_CRCN
from library.loss import DeepSupervisionLoss
from library.trainer import Trainer
from library.predict import generate_predictions
from library.utils import set_seed, save_checkpoint


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print(">>> Step 1: Configuration & Setup")

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Override config constants to make the demo run fast
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.CACHE_DATA = (
        False  # Disable caching to avoid reading/writing large files for just a demo
    )

    # Define a temporary working directory for this demo
    demo_working_dir = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(demo_working_dir, exist_ok=True)

    # Patch the WORKING_DIR in config so Trainer saves/loads from here
    config.WORKING_DIR = demo_working_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Step 2: Data Loading Demonstration")

    # Initialize dataset with a small subset for debugging/demo
    subset_size = 10
    print(f"Initializing GestureDataset with subset_size={subset_size}...")

    train_ds = GestureDataset(
        config.TRAIN_METADATA_PATH,
        is_train=True,
        augment=False,
        subset_size=subset_size,
    )

    assert len(train_ds) == subset_size, f"Dataset should have {subset_size} samples"

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    features, cls_targets, bnd_targets, lengths, mask, sample_ids = batch

    print(f"Batch shapes:")
    print(f"  Features: {features.shape} (Expected: [B, MaxLen, 118])")
    print(f"  Cls Targets: {cls_targets.shape} (Expected: [B, MaxLen])")
    print(f"  Mask: {mask.shape} (Expected: [B, MaxLen])")

    # Assertions
    B, L, D = features.shape
    assert B == config.BATCH_SIZE or B == subset_size
    assert D == 118, f"Input dimension should be 118, got {D}"
    assert cls_targets.shape == (B, L)
    assert mask.shape == (B, L)
    assert len(lengths) == B

    # -------------------------------------------------------------------------
    # 3. Model Forward Pass Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Step 3: Model Forward Pass Demonstration")

    model = GSG_CRCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(features, mask)

    # Verify output structure
    assert isinstance(outputs, dict)
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    # Verify Stage 3 output shapes
    # Output of stage 3 'cls' is (N, L, 21) based on model.py return
    s3_cls = outputs["stage3"]["cls"]
    s3_bnd = outputs["stage3"]["bnd"]

    print(f"Stage 3 Logits Shape: {s3_cls.shape}")

    assert s3_cls.shape == (
        B,
        L,
        config.NUM_CLASSES,
    ), f"Expected (B, L, {config.NUM_CLASSES}), got {s3_cls.shape}"
    assert s3_bnd.shape == (B, L, 1), f"Expected (B, L, 1), got {s3_bnd.shape}"

    # -------------------------------------------------------------------------
    # 4. Loss Computation Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Step 4: Loss Computation Demonstration")

    criterion = DeepSupervisionLoss().to(device)

    cls_targets = cls_targets.to(device)
    bnd_targets = bnd_targets.to(device)

    loss, metrics = criterion(outputs, cls_targets, bnd_targets, mask)

    print(f"Total Loss: {loss.item():.4f}")
    print("Metrics keys:", list(metrics.keys()))

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert "stage3_loss_cls" in metrics

    # -------------------------------------------------------------------------
    # 5. Trainer Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Step 5: Trainer Loop Demonstration")

    # Instantiate Trainer with subset_size
    # We use the patched config.WORKING_DIR
    trainer = Trainer(subset_size=12)

    # Run one training epoch
    print("Running train_epoch(1)...")
    avg_train_loss = trainer.train_epoch(1)
    assert avg_train_loss > 0

    # Run one validation epoch
    print("Running validate_epoch(1)...")
    avg_val_loss, val_score = trainer.validate_epoch(1)

    print(
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Score: {val_score:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Step 6: Inference Demonstration")

    # Save a dummy checkpoint to simulate a trained model
    checkpoint_path = os.path.join(demo_working_dir, "best_model.pth")
    save_checkpoint(
        {
            "epoch": 1,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_score": val_score,
        },
        checkpoint_path,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint failed to save"

    # Define submission output path
    submission_path = os.path.join(demo_working_dir, "submission_demo.csv")

    # Run prediction on a small subset of test data
    generate_predictions(
        checkpoint_path=checkpoint_path,
        output_file=submission_path,
        subset_size=5,
        batch_size=2,
    )

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not created"

    with open(submission_path, "r") as f:
        lines = f.readlines()

    print(f"Submission file created with {len(lines)} lines.")
    if len(lines) > 0:
        print(f"Sample line: {lines[0].strip()}")

    # Check format: SessionID,Labels
    parts = lines[0].strip().split(",")
    assert len(parts) >= 1, "Invalid submission format"
    assert "Sample" in parts[0] or "Session" in parts[0], "First column should be ID"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
