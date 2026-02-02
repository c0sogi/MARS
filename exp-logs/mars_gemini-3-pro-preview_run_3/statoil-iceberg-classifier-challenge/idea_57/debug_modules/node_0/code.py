import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import Config
from library.utils import load_dataset, set_seed
from library.data_loader import get_loaders, get_test_loader
from library.model import WA_IDPH_CNN
from library.train import run_fold


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override for Speed
    # We override the Config parameters to run a minimal version of the task
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.EPOCHS = 2  # Reduce epochs from 75 to 2
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.N_FOLDS = 2  # Setup for 2 folds, though we will only run one
    Config.USE_TTA = False

    # Ensure directories exist (Config.setup is called on import, but good to double check)
    Config.setup()
    set_seed(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    # Load train data (this triggers caching if not already cached)
    X_train, ang_train, y_train, ids_train = load_dataset(
        "train", load_cached_data=True
    )

    # Validate shapes
    # Expected: X -> (N, 3, 75, 75), ang -> (N,), y -> (N,)
    print(f"    Train X shape: {X_train.shape}")
    print(f"    Train angles shape: {ang_train.shape}")
    print(f"    Train labels shape: {y_train.shape}")

    assert len(X_train.shape) == 4, "X_train should be 4D (N, C, H, W)"
    assert X_train.shape[1] == 3, "X_train should have 3 channels (HH, HV, Avg)"
    assert X_train.shape[2:] == (75, 75), "Image dimensions should be 75x75"
    assert len(ang_train) == len(X_train), "Mismatch in angles count"
    assert len(y_train) == len(X_train), "Mismatch in labels count"

    # Load test data
    X_test, ang_test, y_test, ids_test = load_dataset("test", load_cached_data=True)
    print(f"    Test X shape: {X_test.shape}")
    assert y_test is None, "Test labels should be None"

    # 3. DataLoader Verification
    print("\n[3] Verifying DataLoaders...")
    # Get loaders for Fold 0
    train_loader, val_loader = get_loaders(fold_idx=0)

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Angles Shape: {angles.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)
    assert images.dtype == torch.float32
    assert angles.dtype == torch.float32

    # 4. Model Architecture Verification
    print("\n[4] Verifying Model Architecture...")
    model = WA_IDPH_CNN()
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Forward pass
    logits = model(images, angles)
    print(f"    Output Logits Shape: {logits.shape}")

    # Check output shape (Batch_Size,)
    assert logits.shape == (Config.BATCH_SIZE,)

    # 5. Training Loop Demonstration
    print("\n[5] Running Short Training Loop (Fold 0)...")
    # run_fold handles the training loop, validation, and checkpoint saving
    best_val_loss = run_fold(fold_idx=0)

    print(f"    Training completed. Best Val Loss: {best_val_loss:.4f}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"    Checkpoint verified at: {checkpoint_path}")

    # 6. Inference Demonstration
    print("\n[6] Running Inference on Test Set...")

    # Load best model
    model = WA_IDPH_CNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    test_loader = get_test_loader()
    predictions = []

    # Run inference
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(Config.DEVICE)
            angles = angles.to(Config.DEVICE)

            logits = model(images, angles)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    print(f"    Predictions generated: {len(predictions)}")
    print(f"    Prediction Range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    assert len(predictions) == len(
        X_test
    ), "Number of predictions must match test set size"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Probabilities must be in [0, 1]"

    # 7. Submission Generation
    print("\n[7] Generating Submission File...")
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": predictions})

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"    Submission saved to: {submission_path}")

    # Verify file content
    df_check = pd.read_csv(submission_path)
    print(f"    Loaded submission head:\n{df_check.head(3)}")
    assert df_check.shape == (len(X_test), 2)
    assert list(df_check.columns) == ["id", "is_iceberg"]

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
