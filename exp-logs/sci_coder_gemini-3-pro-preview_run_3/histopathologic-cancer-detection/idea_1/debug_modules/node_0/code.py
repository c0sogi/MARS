import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, get_transforms, PathologyDataset
from library.model import get_model
from library.trainer import ModelTrainer


def main():
    print("=== Starting Demonstration Script ===")

    # --- 1. Configuration Overrides for Speed ---
    print("Configuring parameters for rapid execution...")
    # Override Config constants to ensure the script runs quickly (demo mode)
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Reduce workers for small data overhead
    Config.DEBUG_SUBSET_SIZE = 100  # Only use 100 images for this demo

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --- 2. Data Loading and Verification ---
    print("\n[Step 1] Loading Metadata and Creating Datasets...")

    # Load metadata dataframes
    df_train_full = load_metadata("train")
    df_val_full = load_metadata("val")
    df_test_full = load_metadata("test")

    # Subset data for speed
    print(f"Subsetting data to first {Config.DEBUG_SUBSET_SIZE} samples.")
    df_train = df_train_full.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)
    df_val = df_val_full.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)
    df_test = df_test_full.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)

    # Initialize Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Initialize Datasets
    train_dataset = PathologyDataset(df_train, transform=train_transform)
    val_dataset = PathologyDataset(df_val, transform=val_transform)
    test_dataset = PathologyDataset(df_test, transform=val_transform)

    # Verify Dataset Logic
    print("Verifying dataset output...")
    sample_img, sample_label, sample_id = train_dataset[0]

    # Check Image Tensor Shape: (C, H, W) -> (3, 64, 64) based on Config.CROP_SIZE
    expected_shape = (3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        sample_img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {sample_img.shape}"

    # Check Label Type
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    assert sample_label.dtype == torch.float32, "Label should be float32"

    print("Dataset verification successful.")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Model Instantiation and Verification ---
    print("\n[Step 2] Initializing Model...")

    model = get_model()

    # Move to configured device
    model = model.to(Config.DEVICE)

    # Verify Model Output Shape
    print("Verifying model forward pass...")
    dummy_input = torch.randn(2, 3, Config.CROP_SIZE, Config.CROP_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (Batch_Size, Num_Classes) -> (2, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model verification successful.")

    # --- 4. Training Loop ---
    print("\n[Step 3] Starting Training Loop...")

    trainer = ModelTrainer(model, device=Config.DEVICE)

    # Run training (Config.EPOCHS is set to 1)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify Checkpoint Creation
    assert os.path.exists(
        Config.CHECKPOINT_PATH
    ), f"Checkpoint file not found at {Config.CHECKPOINT_PATH}"
    print("Training complete and checkpoint verified.")

    # --- 5. Inference and Submission ---
    print("\n[Step 4] Generating Predictions...")

    trainer.predict(test_loader)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check submission dimensions
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "label",
    ], f"Submission columns mismatch. Expected ['id', 'label'], got {list(df_sub.columns)}"

    # Check value range (probabilities should be between 0 and 1)
    assert (
        df_sub["label"].min() >= 0.0 and df_sub["label"].max() <= 1.0
    ), "Prediction probabilities out of range [0, 1]"

    print("Inference verification successful.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
