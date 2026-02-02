import os
import sys
import numpy as np
import torch
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_loaders, get_test_loader, process_and_cache_data
from library.model import ATSICNN
from library.train import run_kfold, generate_submission


def demo_pipeline():
    # 1. Setup and Configuration Overrides for Speed
    print("--- Setting up Configuration for Demo ---")
    Config.setup()

    # Override Config for rapid execution
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for speed
    Config.NUM_EPOCHS = 1
    Config.NUM_FOLDS = 2
    Config.BATCH_SIZE = 8
    Config.PATIENCE = 1

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Folds: {Config.NUM_FOLDS}")

    # 2. Verify Data Loading and Processing
    print("\n--- Verifying Data Loading ---")
    # This triggers processing and caching
    data_cache = process_and_cache_data(load_cached_data=True)

    # Check cached data shapes
    X_train = data_cache["X_train"]
    assert len(X_train.shape) == 4, "X_train should be 4D (N, C, H, W)"
    assert X_train.shape[1] == 3, "Images should have 3 channels (Band1, Band2, Avg)"
    assert X_train.shape[2] == 75 and X_train.shape[3] == 75, "Images should be 75x75"
    print(f"Data shapes verified. Train size: {X_train.shape[0]}")

    # Get DataLoaders for Fold 0
    train_loader, val_loader = get_loaders(fold_idx=0, load_cached_data=True)

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    # Assertions for batch structure
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Incorrect image batch shape: {images.shape}"
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect angle batch shape: {angles.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label batch shape: {labels.shape}"
    print("DataLoader batch structure verified.")

    # 3. Verify Model Architecture and Forward Pass
    print("\n--- Verifying Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = ATSICNN().to(device)

    # Move batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images_dev, angles_dev)

    # Check output shape (Batch_Size, 1) - Binary Classification Logits
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {outputs.shape}"
    print("Model forward pass successful.")

    # 4. Execute Training Pipeline (Mini-Run)
    print("\n--- Executing Training Pipeline (Mini-Run) ---")
    # This runs the k-fold training loop defined in library/train.py
    # With DEBUG=True and NUM_EPOCHS=1, this should be very fast.
    run_kfold()

    # Verify checkpoints were created
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    for fold in range(Config.NUM_FOLDS):
        ckpt_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {fold} missing at {ckpt_path}"
    print("Training loop completed and checkpoints verified.")

    # 5. Execute Submission Generation
    print("\n--- Generating Submission ---")
    generate_submission()

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"

    # Check values are probabilities (or close to 0/1 if model is confident, but strictly within valid range)
    # Note: Since we only trained for 1 epoch on a tiny subset, predictions might be garbage,
    # but they must be valid floats.
    preds = df_sub["is_iceberg"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Check against test IDs
    _, test_ids = get_test_loader(load_cached_data=True)
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count {len(df_sub)} does not match test set size {len(test_ids)}"

    print("Submission format verified.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    demo_pipeline()
