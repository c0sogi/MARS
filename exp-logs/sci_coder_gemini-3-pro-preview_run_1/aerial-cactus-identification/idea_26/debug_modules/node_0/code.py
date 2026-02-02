import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import CactusDataset, mixup_data
from library.model import get_repvgg_model
from library.engine import train_model
from library.inference import predict_with_calibration


def run_demo():
    # 1. Setup
    print("=== Setting up Demo ===")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    base_dir = "./working/demo_run"
    cache_dir = os.path.join(base_dir, "cache")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    sub_dir = os.path.join(base_dir, "submission")

    # Clean start
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    # 2. Dataset Demonstration
    print("\n=== Demonstrating Dataset ===")
    # Initialize Datasets
    # We use the metadata files generated in the problem description
    train_metadata = "./metadata/train_metadata.csv"
    val_metadata = "./metadata/val_metadata.csv"
    test_metadata = "./metadata/test_metadata.csv"

    # Create Train Dataset
    train_dataset = CactusDataset(
        metadata_file=train_metadata,
        input_dir="./input",
        split="train",
        cache_dir=cache_dir,
        load_cached_data=False,  # Force processing for demo
    )

    # Create Val Dataset
    val_dataset = CactusDataset(
        metadata_file=val_metadata,
        input_dir="./input",
        split="val",
        cache_dir=cache_dir,
        load_cached_data=False,
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Create DataLoaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Validate Batch Structure
    images, labels, qualities = next(iter(train_loader))
    print(
        f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, Qualities: {qualities.shape}"
    )

    # Assertions
    assert images.shape == (batch_size, 3, 32, 32), "Incorrect image tensor shape"
    assert labels.shape == (batch_size,), "Incorrect label tensor shape"
    assert qualities.shape == (batch_size,), "Incorrect quality tensor shape"

    # Demonstrate Mixup
    print("Testing Mixup function...")
    mixed_x, y_a, y_b, q_a, q_b, lam = mixup_data(
        images, labels, qualities, alpha=0.2, device="cpu"
    )
    assert mixed_x.shape == images.shape
    assert y_a.shape == labels.shape

    # 3. Model Demonstration
    print("\n=== Demonstrating Model ===")
    # Initialize RepVGG-A0
    model = get_repvgg_model(model_name="RepVGG-A0", deploy=False)
    model.to(device)

    # Dummy Forward Pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    cls_out, qual_out = model(dummy_input)

    print(f"Model Output Shapes -> Class: {cls_out.shape}, Quality: {qual_out.shape}")

    # Assertions
    assert cls_out.shape == (2, 1), "Classification output shape mismatch"
    assert qual_out.shape == (2, 1), "Quality output shape mismatch"

    # 4. Training Engine Demonstration
    print("\n=== Demonstrating Training Engine ===")

    # Config for short run
    train_config = {
        "epochs": 2,  # Short run for demo
        "swa_start_epoch": 2,  # Trigger SWA at the end
        "patience": 2,
        "save_dir": ckpt_dir,
        "mixup_alpha": 0.2,
        "quality_weight": 0.5,
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config["epochs"]
    )

    # Run Training
    # This handles training loops, validation, SWA, and saving checkpoints
    trained_model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config,
    )

    print(f"Training completed. Final Metrics: {metrics}")

    # Verify checkpoint creation
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    swa_model_path = os.path.join(ckpt_dir, "swa_model.pth")

    if os.path.exists(swa_model_path):
        print(f"SWA model found at {swa_model_path}")
        inference_checkpoint = swa_model_path
    elif os.path.exists(best_model_path):
        print(f"Best model found at {best_model_path}")
        inference_checkpoint = best_model_path
    else:
        raise FileNotFoundError("No model checkpoint was saved during training.")

    # 5. Inference Demonstration
    print("\n=== Demonstrating Inference & Calibration ===")

    submission_path = os.path.join(sub_dir, "submission.csv")

    # The inference function expects a list of fold paths. We pass our single trained model.
    # It performs:
    # 1. Loading model (RepVGG)
    # 2. Reparameterization (Switch to deploy mode)
    # 3. TTA Prediction
    # 4. Quality-based Calibration
    predict_with_calibration(
        fold_paths=[inference_checkpoint],
        test_metadata=test_metadata,
        input_dir="./input",
        output_path=submission_path,
        batch_size=32,
        num_workers=2,
        device=device,
    )

    # 6. Validation of Results
    print("\n=== Validating Submission ===")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df_sub.head())

    # Check dimensions
    expected_rows = 3325  # From dataset info
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id",
        "has_cactus",
    ], "Incorrect columns in submission"

    # Check value ranges
    probs = df_sub["has_cactus"].values
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
