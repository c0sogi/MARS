import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Import provided library modules
from library import config, utils, data_loader, model, train, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(n_train=10, n_val=5, n_test=5):
    """
    Creates mini versions of metadata files to speed up the demonstration.
    """
    print("Creating mini metadata files for demonstration...")

    # Define paths for mini metadata
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # Read original metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subsets
    df_train_mini = df_train.head(n_train)
    df_val_mini = df_val.head(n_val)
    df_test_mini = df_test.head(n_test)

    # Save mini metadata
    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def run_demo():
    # 1. Setup & Configuration
    print("--- 1. Setup & Configuration ---")
    utils.seed_everything(config.SEED)

    # Override config for speed and demo purposes
    config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Create mini datasets
    mini_train, mini_val, mini_test = create_mini_metadata()

    # Update config paths
    config.TRAIN_METADATA_PATH = mini_train
    config.VAL_METADATA_PATH = mini_val
    config.TEST_METADATA_PATH = mini_test

    # Update Hyperparameters for speed
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_FOLDS = 2  # Run 2 folds to demonstrate CV logic
    config.PRETRAINED = False  # Avoid downloading weights
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    print(
        f"Config updated: Epochs={config.EPOCHS}, Folds={config.NUM_FOLDS}, Batch={config.BATCH_SIZE}"
    )

    # 2. Verify Data Loader
    print("\n--- 2. Verifying Data Loader ---")
    # Get loader for fold 0. load_cached_data=False forces processing of our new mini files
    train_loader, val_loader = data_loader.get_dataloaders(
        fold_idx=0, load_cached_data=False
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, 9, 224, 224)
    expected_channels = config.INPUT_CHANNELS
    expected_size = config.IMG_SIZE

    assert images.shape == (
        config.BATCH_SIZE,
        expected_channels,
        expected_size,
        expected_size,
    ), f"Image shape mismatch. Expected {(config.BATCH_SIZE, expected_channels, expected_size, expected_size)}, got {images.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Target shape mismatch. Expected {(config.BATCH_SIZE,)}, got {targets.shape}"

    print("Data Loader verification passed.")

    # 3. Verify Model Architecture
    print("\n--- 3. Verifying Model Architecture ---")
    net = model.ACWIVNet(pretrained=False)
    net.to(config.DEVICE)
    net.eval()

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        images = images.to(config.DEVICE)
        output = net(images)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model architecture verification passed.")

    # 4. Run Training Pipeline
    print("\n--- 4. Running Training Pipeline (2 Folds) ---")
    # This calls train.run_fold for range(NUM_FOLDS)
    train.train_all_folds()

    # Verify checkpoints exist
    for fold in range(config.NUM_FOLDS):
        ckpt_path = os.path.join(config.WORKING_DIR, f"best_model_fold{fold}.pth")
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {fold} missing at {ckpt_path}"
        print(f"Checkpoint verified: {ckpt_path}")

    # 5. Run Inference Pipeline
    print("\n--- 5. Running Inference Pipeline ---")
    # This loads the checkpoints we just trained and predicts on the mini test set
    inference.predict(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print("Submission head:")
    print(df_sub.head())

    # Verify submission content
    # We used 5 test subjects in create_mini_metadata
    assert len(df_sub) == 5, f"Submission should have 5 rows, got {len(df_sub)}"
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Missing columns in submission."

    print("Inference pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
