import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import process_and_cache_data, get_cv_loaders, get_test_loader
from library.model import DPDB_NBA_CNN
from library.train import run_kfold_training, generate_submission


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Environment and Overrides for Speed
    # We override Config parameters to run a "mini" version of the task
    # This ensures the script finishes quickly while testing all components.

    # Create a separate working directory for the demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_SUBMISSION_DIR = "./working/demo_submission"

    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    if os.path.exists(DEMO_SUBMISSION_DIR):
        shutil.rmtree(DEMO_SUBMISSION_DIR)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Patch Config class dynamically
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the demo directory
    Config.CACHE_X_TRAIN = os.path.join(DEMO_WORKING_DIR, "X_train.npy")
    Config.CACHE_Y_TRAIN = os.path.join(DEMO_WORKING_DIR, "y_train.npy")
    Config.CACHE_ANGLE_TRAIN = os.path.join(DEMO_WORKING_DIR, "angle_train.npy")
    Config.CACHE_X_TEST = os.path.join(DEMO_WORKING_DIR, "X_test.npy")
    Config.CACHE_ANGLE_TEST = os.path.join(DEMO_WORKING_DIR, "angle_test.npy")
    Config.CACHE_ID_TEST = os.path.join(DEMO_WORKING_DIR, "id_test.npy")

    # Reduce compute requirements for demo
    Config.N_FOLDS = 2  # Run only 2 folds instead of 5
    Config.NUM_EPOCHS = 1  # Run only 1 epoch per fold
    Config.BATCH_SIZE = 4  # Small batch size
    Config.PATIENCE = 1  # Minimal patience

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration patched for demo execution.")

    # 2. Data Processing and Loading Demo
    print("\n--- Testing Data Processing ---")
    # This function reads the JSONs, processes bands, imputes angles, and saves .npy files
    # We force `load_cached_data=False` to ensure processing logic runs.
    X_train, y_train, angle_train, X_test, id_test, angle_test = process_and_cache_data(
        load_cached_data=False
    )

    # Assertions to verify data integrity
    print(f"Train Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    assert X_train.shape[1:] == (
        75,
        75,
        3,
    ), f"Expected (N, 75, 75, 3), got {X_train.shape}"
    assert X_test.shape[1:] == (
        75,
        75,
        3,
    ), f"Expected (N, 75, 75, 3), got {X_test.shape}"
    assert (
        len(X_train) == len(y_train) == len(angle_train)
    ), "Training data lengths mismatch"
    assert len(X_test) == len(id_test) == len(angle_test), "Test data lengths mismatch"
    assert not np.isnan(angle_train).any(), "NaN found in training angles"
    assert not np.isnan(angle_test).any(), "NaN found in test angles"

    # 3. DataLoader Demo
    print("\n--- Testing Data Loaders ---")
    # Get loaders for Fold 0, limiting samples to 20 for speed
    train_loader, val_loader = get_cv_loaders(
        fold_idx=0, load_cached_data=True, max_samples=20
    )

    # Fetch one batch to verify tensor shapes
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Expected: (Batch, Channels, H, W) -> (4, 3, 75, 75)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"

    # 4. Model Instantiation and Forward Pass Demo
    print("\n--- Testing Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = DPDB_NBA_CNN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output should be (Batch, 1)"

    # Verify loss calculation works
    criterion = nn.BCEWithLogitsLoss()
    labels = labels.to(device).unsqueeze(1)  # (Batch, 1)
    loss = criterion(outputs, labels)
    print(f"Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Training Loop Demo
    print("\n--- Testing Training Loop (Miniature) ---")
    # Run training for 2 folds, 1 epoch each, using only 32 samples per fold
    fold_scores = run_kfold_training(max_samples=32, num_epochs=1)

    assert (
        len(fold_scores) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} scores, got {len(fold_scores)}"

    # Check if checkpoints were created
    for fold in range(Config.N_FOLDS):
        ckpt_path = Config.get_checkpoint_path(fold)
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {fold} not found at {ckpt_path}"
        print(f"Checkpoint verified: {ckpt_path}")

    # 6. Submission Generation Demo
    print("\n--- Testing Submission Generation ---")
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head()}")

    # Check submission format
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission"
    assert len(df_sub) == len(
        X_test
    ), f"Submission length {len(df_sub)} != Test set length {len(X_test)}"

    # Check probability range
    probs = df_sub["is_iceberg"].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of [0, 1] range"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
