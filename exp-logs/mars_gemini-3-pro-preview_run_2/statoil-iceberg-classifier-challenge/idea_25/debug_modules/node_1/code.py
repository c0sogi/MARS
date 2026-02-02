import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import from the provided library
from library.config import Config, set_seed
from library.utils import get_logger
from library.data_loader import load_data, get_dataloaders, get_test_loader
from library.model import SIWBN
from library.train import run_fold, validate


def run_demo():
    print("Initializing Demo...")

    # ==========================================
    # 1. CONFIGURATION OVERRIDE FOR SPEED
    # ==========================================
    # We patch the Config class to run a lightweight version of the pipeline.
    print("Patching Config for fast execution...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SIZE = 20  # Use only 20 samples
    Config.NUM_FOLDS = 2  # We will only run Fold 0, but setup for 2
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "processed_data_debug.npz"
    )

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. DATA LOADING DEMONSTRATION
    # ==========================================
    print("\n[Step 1] Loading Data (Debug Mode)...")
    # load_data with debug=True returns a small slice of the data
    train_data, test_data = load_data(debug=True, load_cached_data=False)

    X_train, inc_train, y_train, train_ids = train_data
    X_test, inc_test, test_ids = test_data

    # Validation of Data Shapes
    print(f"  Train Data Shape: {X_train.shape}")
    print(f"  Test Data Shape:  {X_test.shape}")

    assert (
        len(X_train) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} train samples, got {len(X_train)}"
    assert X_train.shape[1:] == (
        75,
        75,
        3,
    ), f"Expected image shape (75, 75, 3), got {X_train.shape[1:]}"
    assert len(y_train) == Config.DEBUG_SIZE, "Label count mismatch"

    print("  Data loaded and verified successfully.")

    # ==========================================
    # 3. DATALOADER DEMONSTRATION
    # ==========================================
    print("\n[Step 2] Creating Dataloaders for Fold 0...")
    # Get loaders for the first fold
    train_loader, val_loader = get_dataloaders(fold_index=0, train_data=train_data)

    # Fetch a single batch to verify structure
    images, angles, labels = next(iter(train_loader))

    print(f"  Batch Images Shape: {images.shape}")  # Should be (Batch, 3, 75, 75)
    print(f"  Batch Angles Shape: {angles.shape}")
    print(f"  Batch Labels Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect batch image shape"
    assert angles.shape[0] == Config.BATCH_SIZE, "Incorrect batch angle shape"
    assert labels.shape[0] == Config.BATCH_SIZE, "Incorrect batch label shape"

    print("  Dataloaders verified successfully.")

    # ==========================================
    # 4. MODEL INSTANTIATION & VERIFICATION
    # ==========================================
    print("\n[Step 3] Initializing SIWBN Model...")
    device = Config.DEVICE
    model = SIWBN().to(device)

    # Run a forward pass with the batch fetched earlier
    print("  Running forward pass check...")
    model.eval()
    with torch.no_grad():
        images_dev = images.to(device)
        angles_dev = angles.to(device)
        outputs = model(images_dev, angles_dev)

    print(f"  Output Shape: {outputs.shape}")
    print(f"  Output Values: {outputs.flatten().cpu().numpy()}")

    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (outputs >= 0).all() and (
        outputs <= 1
    ).all(), "Model output not in probability range [0, 1]"

    print("  Model forward pass verified successfully.")

    # ==========================================
    # 5. TRAINING LOOP DEMONSTRATION
    # ==========================================
    print("\n[Step 4] Running Training Loop (Fold 0)...")
    # run_fold encapsulates the training loop, validation, and early stopping
    # We use a custom logger to see output in stdout
    logger = get_logger(os.path.join(Config.WORKING_DIR, "demo.log"))

    trained_model, best_loss = run_fold(
        fold_index=0, train_data=train_data, logger=logger
    )

    print(f"  Training completed. Best Validation Loss: {best_loss:.4f}")

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"  Checkpoint verified at {checkpoint_path}")

    # ==========================================
    # 6. INFERENCE DEMONSTRATION
    # ==========================================
    print("\n[Step 5] Running Inference on Test Set...")
    # Create test loader
    test_loader, test_ids_out = get_test_loader(train_data, test_data)

    trained_model.eval()
    predictions = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)
            outputs = trained_model(images, angles)
            predictions.extend(outputs.view(-1).cpu().numpy())

    predictions = np.array(predictions)

    print(f"  Generated {len(predictions)} predictions.")
    print(f"  Sample predictions: {predictions[:5]}")

    assert len(predictions) == len(
        test_ids_out
    ), "Number of predictions does not match number of test IDs"

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_ids_out, "is_iceberg": predictions})

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"  Submission saved to {submission_path}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
