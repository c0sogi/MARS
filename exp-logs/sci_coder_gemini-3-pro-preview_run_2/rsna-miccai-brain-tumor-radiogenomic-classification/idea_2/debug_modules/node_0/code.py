import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import ModalityGroupedEfficientNet
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Glioblastoma Classification Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config parameters for speed and deterministic behavior
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Skip weight download for speed/offline safety
    Config.DEBUG = True  # Use subset of data

    # Ensure working directory is clean for the demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print("    Configuration updated: Epochs=1, Batch=4, Pretrained=False")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading and Processing...")

    # Initialize dataloaders with debug=True to load a small subset
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch from the training loader
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"    Batch fetched successfully.")
    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Label Batch Shape: {labels.shape}")

    # Assertions to verify data integrity
    # Expected shape: (Batch, Channels, Height, Width)
    # Channels = 4 modalities * 3 slices = 12
    expected_channels = 12
    assert images.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, expected_channels, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"

    assert (
        labels.shape[0] == Config.BATCH_SIZE
    ), f"Incorrect label batch size. Expected {Config.BATCH_SIZE}, got {labels.shape[0]}"

    assert images.dtype == torch.float32, "Images should be float32 tensor"

    print("    Data integrity checks passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = ModalityGroupedEfficientNet()
    model.to(device)

    # Move batch to device
    images = images.to(device)

    # Perform forward pass
    logits = model(images)

    print(f"    Forward pass output shape: {logits.shape}")

    # Assertions for model output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    assert logits.requires_grad, "Model output should require gradients for training"

    print("    Model architecture checks passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, device)

    # Run training
    # This will train for 1 epoch (as set in config override) and validate
    trainer.fit(epochs=Config.EPOCHS, patience=1)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint was not created at {Config.MODEL_PATH}"

    print(f"    Training complete. Checkpoint found at: {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference and Submission Generation...")

    # Load the best model state
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    test_ids = []

    print("    Running inference on test set...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)

            # Get corresponding IDs from the dataset
            # The dataset was shuffled=False for test loader
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_ids = test_loader.dataset.df.iloc[start_idx:end_idx][
                "BraTS21ID"
            ].values
            test_ids.extend(batch_ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    print(f"    Generated predictions for {len(submission_df)} samples.")
    print("    Sample predictions:")
    print(submission_df.head())

    # Verify submission format
    assert "BraTS21ID" in submission_df.columns
    assert "MGMT_value" in submission_df.columns
    assert len(submission_df) > 0
    assert submission_df["MGMT_value"].min() >= 0.0
    assert submission_df["MGMT_value"].max() <= 1.0

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
