import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.model import WBMGNet
from library.data_loader import (
    load_and_process_data,
    get_fold_dataloaders,
    get_test_dataloader,
)
from library.train_eval import train_fold, predict


def run_demo():
    print("Starting Demo Script...")

    # 1. Setup Configuration for Demo (Speed Optimization)
    print("\n[Step 1] Configuring for Demo Mode...")
    # Override Config for fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size

    # Use a specific cache file for debug to avoid overwriting full data cache
    Config.CACHE_FILE = "processed_data_debug.npz"
    Config.CACHE_PATH = os.path.join(Config.WORKING_DIR, Config.CACHE_FILE)

    # Create necessary directories
    Config.create_directories()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    # Force reload (load_cached_data=False) to ensure we process the raw JSON
    # and apply the DEBUG slicing logic.
    (
        train_imgs,
        train_angles,
        train_labels,
        train_ids,
        test_imgs,
        test_angles,
        test_ids,
    ) = load_and_process_data(load_cached_data=False)

    # Assertions for Data Dimensions
    assert (
        len(train_imgs) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_imgs)}"
    assert (
        len(test_imgs) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: expected {Config.DEBUG_SAMPLE_SIZE}, got {len(test_imgs)}"
    # Check image shape: (N, 3, 75, 75)
    assert train_imgs.shape[1:] == (
        3,
        75,
        75,
    ), f"Image shape mismatch: {train_imgs.shape}"
    # Check for NaNs in image data
    assert not np.isnan(train_imgs).any(), "NaNs found in training images"

    print("Data Loading Verified.")
    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Test Images Shape: {test_imgs.shape}")

    # 3. DataLoader & Preprocessing Verification
    print("\n[Step 3] Verifying DataLoaders and Fold Logic...")
    fold_idx = 0
    # Now we can load from cache since Step 2 created it
    train_loader, val_loader, stats = get_fold_dataloaders(
        fold_idx, load_cached_data=True
    )

    # Verify Stats dictionary structure
    assert "min_vals" in stats
    assert "max_vals" in stats
    assert "angle_mean" in stats

    # Verify Batch Shapes from DataLoader
    images_batch, angles_batch, labels_batch = next(iter(train_loader))
    assert images_batch.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Batch image shape mismatch: {images_batch.shape}"
    assert angles_batch.shape == (
        Config.BATCH_SIZE,
    ), f"Batch angle shape mismatch: {angles_batch.shape}"
    assert labels_batch.shape == (
        Config.BATCH_SIZE,
    ), f"Batch label shape mismatch: {labels_batch.shape}"

    print("DataLoaders Verified.")

    # 4. Model Instantiation & Forward Pass
    print("\n[Step 4] Verifying Model Architecture...")
    model = WBMGNet().to(device)

    # Create dummy input to test forward pass
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_ang = torch.randn(4).to(device)

    # Perform forward pass
    output = model(dummy_img, dummy_ang)

    # Check output shape (B, 1) for binary classification logits
    assert output.shape == (4, 1), f"Model output shape mismatch: {output.shape}"
    print("Model Forward Pass Verified.")

    # 5. Training Loop Demonstration
    print("\n[Step 5] Running Training Loop (Fold 0)...")
    # train_fold handles the training loop, validation, and checkpoint saving
    trained_model, history = train_fold(fold_idx, train_loader, val_loader, device)

    # Verify History content
    assert len(history["train_loss"]) > 0
    assert len(history["val_loss"]) > 0
    print(f"Training finished. Final Val Acc: {history['val_acc'][-1]:.4f}")

    # Verify Checkpoint Existence
    checkpoint_path = Config.get_checkpoint_path(fold_idx)
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Checkpoint verified at {checkpoint_path}")

    # 6. Inference Demonstration
    print("\n[Step 6] Running Inference on Test Set...")
    # Get test loader using stats from the training fold (normalization/imputation)
    test_loader, test_ids_loader = get_test_dataloader(stats, load_cached_data=True)

    # Run prediction
    predictions = predict(trained_model, test_loader, device)

    # Verify Predictions
    assert len(predictions) == Config.DEBUG_SAMPLE_SIZE
    # Check probability range
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Inference Verified.")
    print(f"Sample Predictions: {predictions[:5]}")

    # 7. Submission File Generation
    print("\n[Step 7] Generating Sample Submission...")
    submission_df = pd.DataFrame({"id": test_ids_loader, "is_iceberg": predictions})

    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)

    assert os.path.exists(demo_submission_path)
    print(f"Submission saved to {demo_submission_path}")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
