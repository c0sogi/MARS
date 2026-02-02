import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import AppleConvNeXt
from library.engine import train_model
from library.inference import run_inference


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Apple Disease Detection Demo...")
    seed_everything(42)

    # Patch Config for a fast demonstration run
    # We modify class attributes directly to influence the behavior of library components
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.IMG_SIZE = 256  # Reduced from 384 for speed
    Config.NUM_WORKERS = 2

    # Define demo-specific paths to avoid overwriting production files
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean and create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Configuration patched. Output directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Dataset & DataLoader Verification
    # ==========================================
    print("\n--- Verifying Data Pipeline ---")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Create tiny subsets for speed
    train_subset = train_df.iloc[:50].reset_index(drop=True)
    val_subset = val_df.iloc[:20].reset_index(drop=True)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # Initialize Datasets
    train_dataset = AppleDataset(train_subset, transforms=get_transforms(data="train"))
    val_dataset = AppleDataset(val_subset, transforms=get_transforms(data="valid"))

    # Verify single item retrieval
    img, target = train_dataset[0]

    # Assertions for shape and type
    assert isinstance(img, torch.Tensor), "Dataset output image must be a Tensor."
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Target shape mismatch. Expected ({Config.NUM_CLASSES},), got {target.shape}"

    print("Dataset shapes verified successfully.")

    # ==========================================
    # 3. Model Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Initialize model (pretrained=True to verify download/loading capability)
    model = AppleConvNeXt(pretrained=True)
    model.to(device)

    # Dummy forward pass
    dummy_batch_size = 2
    dummy_input = torch.randn(dummy_batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        device
    )

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        dummy_batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected ({dummy_batch_size}, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model forward pass verified successfully.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # Prepare Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Execute Training
    # We explicitly pass num_epochs=1 to override any default args bound at import time
    best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=1,
    )

    print(f"Training complete. Best Validation F1: {best_f1}")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Checkpoint file verified.")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\n--- Executing Inference ---")

    # Run inference on a subset of test data (max_samples=20)
    # This uses the 'best_model.pth' we just trained
    run_inference(max_samples=20)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Schema
    expected_cols = ["image", "labels"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(submission_df.columns)}"
    assert len(submission_df) > 0, "Submission file is empty."

    # Check Content Format (Space delimited labels)
    sample_label = submission_df.iloc[0]["labels"]
    assert isinstance(sample_label, str), "Labels must be strings."

    print("Submission file verified successfully.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
