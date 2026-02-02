import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device, ensure_dir
from library.data_loader import GestureDataset, get_dataloaders
from library.model import DSG_CRCN
from library.loss import DSG_Loss
from library.trainer import Trainer
from library.inference import Predictor


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Set fixed seed for reproducibility
    set_seed(42)

    # Override Config parameters for a quick demo run
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.CACHE_DIR = "./working/demo_run/cache"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"

    # Ensure working directories exist
    ensure_dir(Config.CACHE_DIR)
    ensure_dir(Config.SUBMISSION_DIR)

    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    device = get_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading...")

    # Load a tiny subset of the training data (limit=10 samples)
    # We set load_cached_data=False to force processing from raw files for demonstration
    train_ds = GestureDataset(split="train", load_cached_data=False, limit=10)

    print(f"Dataset size: {len(train_ds)}")
    assert len(train_ds) == 10, "Dataset limit did not work."

    # Inspect a single item
    features, labels, boundaries = train_ds[0]
    print(f"Sample 0 Feature Shape: {features.shape}")  # (T, InputDim)
    print(f"Sample 0 Label Shape: {labels.shape}")  # (T,)

    # Verify input dimension matches Config
    assert (
        features.shape[1] == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {features.shape[1]}"

    # Test DataLoader and Collate function
    # We use the helper function but force it to use our limited dataset logic
    # by manually creating a loader here for verification
    from library.data_loader import collate_fn
    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    b_features = batch["features"].to(device)
    b_mask = batch["mask"].to(device)
    b_labels = batch["labels"].to(device)
    b_boundaries = batch["boundaries"].to(device)

    print(f"Batch Features Shape: {b_features.shape}")  # (B, T_max, D)
    assert b_features.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."

    # -------------------------------------------------------------------------
    # 3. Model Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Forward Pass...")

    model = DSG_CRCN().to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(b_features, b_mask)

    # Check output structure
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

    # Check shape of final classification output: (B, T, NumClasses)
    s3_cls = outputs["stage3_cls"]
    print(f"Stage 3 Output Shape: {s3_cls.shape}")
    assert s3_cls.shape[0] == Config.BATCH_SIZE
    assert s3_cls.shape[2] == Config.NUM_CLASSES

    # -------------------------------------------------------------------------
    # 4. Loss Computation Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Loss Computation...")

    criterion = DSG_Loss().to(device)
    targets = {"labels": b_labels, "boundaries": b_boundaries, "mask": b_mask}

    # We need gradients for loss verification usually, but here just checking calculation
    loss, metrics = criterion(outputs, targets)

    print(f"Total Loss: {loss.item():.4f}")
    print("Metrics:", metrics)

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive."

    # -------------------------------------------------------------------------
    # 5. Trainer Integration (Training Loop)
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    # Note: Trainer internally calls get_dataloaders.
    # Since we want to use limited datasets for speed, we need to monkey-patch
    # or ensure get_dataloaders respects our desire for speed.
    # The provided Trainer class doesn't accept dataset arguments, it calls get_dataloaders().
    # The provided get_dataloaders function creates full datasets.
    # To make this fast without modifying library code, we will rely on the fact
    # that we can't easily inject the limited dataset into the Trainer instance
    # without modifying the Trainer or get_dataloaders.
    # HOWEVER, for this demo, we can manually overwrite the loaders in the trainer instance.

    trainer = Trainer()

    # Overwrite loaders with limited ones for speed
    train_ds_limit = GestureDataset("train", load_cached_data=True, limit=20)
    val_ds_limit = GestureDataset("val", load_cached_data=True, limit=10)

    trainer.train_loader = DataLoader(
        train_ds_limit,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    trainer.val_loader = DataLoader(
        val_ds_limit, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Run training
    trainer.fit()

    # Verify checkpoint creation
    expected_ckpt = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"
    print("Training complete and checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Verification
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference & Generating Submission...")

    # Initialize Predictor
    # It will automatically look for the best_model.pth in SUBMISSION_DIR or CACHE_DIR
    predictor = Predictor(batch_size=Config.BATCH_SIZE)

    # Run prediction on a limited test set
    # We use a small limit to ensure it finishes quickly
    sample_ids, predictions = predictor.predict_dataset(dataset_split="test", limit=5)

    print(f"Generated predictions for {len(sample_ids)} samples.")
    print(f"Sample 0 ID: {sample_ids[0]}")
    print(f"Sample 0 Prediction: {predictions[0]}")

    # Generate submission file
    output_csv = "demo_submission.csv"
    predictor.generate_submission(output_file=output_csv)

    submission_path = os.path.join(Config.SUBMISSION_DIR, output_csv)
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify file content format
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"Submission file head ({len(lines)} lines):")
        for line in lines[:3]:
            print(line.strip())

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
