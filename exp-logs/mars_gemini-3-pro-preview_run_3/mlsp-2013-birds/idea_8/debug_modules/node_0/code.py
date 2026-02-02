import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import get_dataloaders, BirdDataset
from library.model import MultiViewResNet
from library.train import run_fold, train_one_epoch, validate


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup Configuration for Demonstration
    # We override default Config values to ensure the script runs quickly (within minutes)
    print("\n[Step 1] Configuring Environment...")
    Config.PROJECT_NAME = "demo_project"
    Config.DEBUG = True  # Use subset of data
    Config.DEBUG_SAMPLES = 20  # Only 20 samples
    Config.EPOCHS = 2  # Only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.NUM_FOLDS = 2  # Only need to demonstrate one fold

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Demonstrate Data Loading
    print("\n[Step 2] Demonstrating Data Loading...")

    # Get DataLoaders for Fold 0
    # This triggers cache creation/loading and dataset splitting
    train_loader, val_loader, test_loader = get_dataloaders(
        fold_idx=0, load_cached_data=False, debug=Config.DEBUG
    )

    # Verify Train Loader
    print(f"Train Loader length: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader should not be empty"

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(
        f"Batch Image Shape: {images.shape}"
    )  # Expected: (B, 3, 3, 224, 224) -> (Batch, Tiles, Channels, H, W)
    print(f"Batch Label Shape: {labels.shape}")  # Expected: (B, 19)

    # Assertions for Data
    assert (
        images.dim() == 5
    ), f"Expected 5D input tensor (B, Tiles, C, H, W), got {images.dim()}D"
    assert (
        images.shape[1] == Config.NUM_TILES
    ), f"Expected {Config.NUM_TILES} tiles, got {images.shape[1]}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, got {labels.shape[1]}"

    print("Data Loading Verification Passed.")

    # 3. Demonstrate Model Architecture
    print("\n[Step 3] Demonstrating Model Architecture...")

    model = MultiViewResNet()
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)

    # Forward pass
    # We use no_grad because we are just testing the forward pass structure
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for Model
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape ({Config.BATCH_SIZE}, {Config.NUM_CLASSES}), got {outputs.shape}"
    assert torch.isfinite(outputs).all(), "Model output contains NaNs or Infs"

    print("Model Verification Passed.")

    # 4. Demonstrate Training Loop (Full Fold Execution)
    print("\n[Step 4] Demonstrating Training Loop (Fold 0)...")

    # run_fold handles model init, optimizer, loop, validation, and saving
    # We run it for the configured small number of epochs
    run_fold(fold_idx=0, debug=Config.DEBUG)

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fold_0_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully saved at: {checkpoint_path}")
    else:
        # It's possible validation AUC didn't improve if random init was lucky,
        # but with 2 epochs and pretrained weights, it usually saves at least once.
        print("Warning: No checkpoint saved (Validation AUC might not have improved).")

    # 5. Demonstrate Inference and Submission Generation
    print("\n[Step 5] Demonstrating Inference and Submission...")

    # Load the best model (or a fresh one if checkpoint missing for some reason)
    model = MultiViewResNet()
    model.to(Config.DEVICE)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        print("Loaded best model from checkpoint.")
    else:
        print("Using initialized model for inference demonstration.")

    model.eval()

    # Run inference on Test Loader
    # Re-instantiate test loader to ensure we start from beginning
    _, _, test_loader = get_dataloaders(
        fold_idx=0, load_cached_data=True, debug=Config.DEBUG
    )

    # Accessing the dataset from the loader to get metadata (rec_id)
    test_df = test_loader.dataset.df

    all_probs = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Verify alignment
    # Since we used DEBUG, the test_df is subsampled.
    # all_probs length should match test_df length.
    print(f"Number of test predictions: {len(all_probs)}")
    print(f"Number of test metadata entries: {len(test_df)}")
    assert len(all_probs) == len(
        test_df
    ), "Mismatch between predictions and metadata length"

    # Create Submission DataFrame
    # Format: Id,Probability
    # Id = rec_id * 100 + species_id

    submission_rows = []
    for idx, row in test_df.iterrows():
        rec_id = row["rec_id"]
        row_probs = all_probs[idx]  # Shape (19,)

        for species_id, prob in enumerate(row_probs):
            submission_id = int(rec_id * 100 + species_id)
            submission_rows.append({"Id": submission_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save Submission
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission file generated at: {submission_path}")
    print("Head of submission file:")
    print(submission_df.head())

    # Final Validation of Submission File
    assert os.path.exists(submission_path), "Submission file was not created"
    loaded_sub = pd.read_csv(submission_path)
    assert list(loaded_sub.columns) == [
        "Id",
        "Probability",
    ], "Submission columns mismatch"
    assert (
        len(loaded_sub) == len(test_df) * Config.NUM_CLASSES
    ), "Submission row count mismatch"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
