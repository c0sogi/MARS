import os
import torch
import numpy as np
import pandas as pd
import sys

# Import library modules
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data_loader as data_loader
import library.train_eval as train_eval


def run_demo():
    print("=== Starting Demonstration ===")

    # 1. Setup and Monkey Patching for Speed
    # We need to update constants in the imported modules because they were imported
    # using 'from library.config import ...' style in the library files.

    print("Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a subset of data
    config.DEBUG = True
    data_loader.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 50
    data_loader.DEBUG_SUBSET_SIZE = 50

    # Reduce Epochs and Batch Size
    config.NUM_EPOCHS = 1
    train_eval.NUM_EPOCHS = 1

    config.BATCH_SIZE = 4
    data_loader.BATCH_SIZE = 4

    # Reduce Folds for the demo (we will only train fold 0, but submission loops over folds)
    config.NUM_FOLDS = 2
    train_eval.NUM_FOLDS = 2
    data_loader.NUM_FOLDS = 2

    # Ensure directories exist
    config.setup_directories()

    # Set seeds
    utils.set_seed(config.SEED)

    # 2. Data Loader Demonstration
    print("\n--- Testing Data Loader ---")

    # Get loaders for Fold 0
    # Note: This will trigger cache generation if not present
    train_loader, val_loader = data_loader.get_data_loaders(
        fold_idx=0, load_cached_data=True
    )

    # Fetch one batch from train loader
    images, angles, labels = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Angles Shape: {angles.shape}")
    print(f"Train Batch - Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert angles.shape == (config.BATCH_SIZE,), "Incorrect angle batch shape"
    assert labels.shape == (config.BATCH_SIZE,), "Incorrect label batch shape"
    assert not torch.isnan(images).any(), "Images contain NaNs"
    assert not torch.isnan(angles).any(), "Angles contain NaNs (Imputation failed)"

    print("Data Loader verification passed.")

    # 3. Model Demonstration
    print("\n--- Testing Model Architecture ---")

    net = model_lib.MCICNN().to(config.DEVICE)

    # Check parameter count
    num_params = utils.count_parameters(net)
    print(f"Model Parameter Count: {num_params}")
    assert num_params > 0, "Model has no parameters"

    # Forward pass verification
    # Move batch to device
    images = images.to(config.DEVICE)
    angles = angles.to(config.DEVICE)

    logits = net(images, angles)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), "Output shape mismatch. Expected (B, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Testing Training Loop (Fold 0) ---")

    # Train for 1 epoch (as configured above)
    best_val_loss = train_eval.train_fold(fold_idx=0, load_cached_data=True)

    print(f"Training completed. Best Val Loss: {best_val_loss}")

    # Check if checkpoint exists
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model_fold_0.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    print("Training loop verification passed.")

    # 5. Submission Demonstration
    print("\n--- Testing Submission Generation ---")

    # To make generate_submission work without training all folds,
    # we need to simulate the existence of other fold checkpoints or accept that it skips them.
    # The provided generate_submission function skips missing checkpoints with a warning.
    # However, to test the averaging logic, let's copy fold 0 checkpoint to fold 1.

    ckpt_path_fold_1 = os.path.join(config.CHECKPOINT_DIR, "model_fold_1.pth")

    # Load fold 0 state dict and save as fold 1
    state_dict = torch.load(ckpt_path)
    torch.save(state_dict, ckpt_path_fold_1)
    print("Simulated checkpoint for Fold 1 created.")

    # Generate submission
    train_eval.generate_submission(load_cached_data=True)

    # Verify output
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(df_sub)}")
    print(df_sub.head())

    # Assertions on submission
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Missing columns in submission"

    # In Debug mode, we only load a subset of test data.
    # The test.json has 1 entry in the provided description, but the sample_submission has 321.
    # The code loads ids from test.json.
    # We just need to ensure the dataframe is not empty.
    assert len(df_sub) > 0, "Submission dataframe is empty"

    # Check probability range
    probs = df_sub["is_iceberg"]
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    print("Submission generation verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
