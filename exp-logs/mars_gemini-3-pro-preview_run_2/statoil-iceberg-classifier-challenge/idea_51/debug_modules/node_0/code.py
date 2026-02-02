import os
import sys
import numpy as np
import pandas as pd
import torch
import logging

# Import the provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    print("Setting up environment and overriding config for demo speed...")

    # Set seeds for reproducibility
    utils.seed_everything(config.SEED)

    # Override config parameters for a fast demonstration run
    config.NUM_EPOCHS = 2  # Train for only 2 epochs
    config.DEBUG_MAX_SAMPLES = (
        100  # Use only 100 samples to speed up data loading/training
    )
    config.BATCH_SIZE = 16  # Smaller batch size for the small subset
    config.NUM_FOLDS = 2  # Setup for 2 folds (though we'll only run one)
    config.PATIENCE = 1  # Short patience for early stopping

    # Ensure working directory exists (handled by config, but good practice)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. DATA PROCESSING
    # ==========================================
    print("Processing data...")
    # This loads raw JSONs, converts bands to images, handles incidence angles, and caches results.
    # Note: process_and_cache_data processes the FULL dataset first, then get_dataloaders subsets it.
    # Since the dataset is relatively small (1.6k train), processing full data is acceptable (~seconds).
    data_dict, scaler = data.process_and_cache_data(load_cached_data=False)

    # VALIDATION: Check data dictionary keys and shapes
    required_keys = ["X_train", "y_train", "inc_train", "X_test", "inc_test"]
    for key in required_keys:
        if key not in data_dict:
            raise AssertionError(f"Missing key {key} in processed data dictionary")

    print(f"Full Training Data Shape: {data_dict['X_train'].shape}")
    print(f"Full Test Data Shape: {data_dict['X_test'].shape}")

    # Assertions for data integrity
    assert data_dict["X_train"].shape[1:] == (75, 75, 3), "Incorrect image dimensions"
    assert not np.isnan(data_dict["X_train"]).any(), "NaNs found in training images"

    # ==========================================
    # 3. DATALOADERS SETUP
    # ==========================================
    print("Setting up dataloaders for Fold 0...")
    # This will apply the DEBUG_MAX_SAMPLES limit
    train_loader, val_loader, test_loader = data.get_dataloaders(
        fold_idx=0, data=data_dict, scaler=scaler
    )

    # VALIDATION: Check batch shapes
    sample_imgs, sample_incs, sample_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_imgs.shape}")
    print(f"Batch Inc Angle Shape: {sample_incs.shape}")
    print(f"Batch Label Shape: {sample_labels.shape}")

    assert sample_imgs.shape == (
        config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect batch image shape"
    assert sample_incs.shape == (
        config.BATCH_SIZE,
        1,
    ), "Incorrect incidence angle shape"

    # ==========================================
    # 4. MODEL INITIALIZATION & VERIFICATION
    # ==========================================
    print("Initializing model...")
    net = model.TKA_WBN()
    net.to(config.DEVICE)

    # VALIDATION: Run a dummy forward pass
    print("Running dummy forward pass...")
    net.eval()
    with torch.no_grad():
        dummy_imgs = sample_imgs.to(config.DEVICE)
        dummy_incs = sample_incs.to(config.DEVICE)
        output = net(dummy_imgs, dummy_incs)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model output not in [0, 1] range (Sigmoid check)"

    # ==========================================
    # 5. TRAINING (Fold 0)
    # ==========================================
    print("Starting training for Fold 0...")
    # train.run_fold handles optimizer, criterion, loop, and saving best weights
    best_weights = train.run_fold(fold_idx=0, data_dict=data_dict, scaler=scaler)

    # Load best weights into model for inference
    net.load_state_dict(best_weights)

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("Running inference on Test set...")
    net.eval()
    predictions = []

    # We need to iterate over the test_loader (which respects DEBUG_MAX_SAMPLES)
    # Note: In a real submission, we would run on the full test set.
    # Here we demonstrate the mechanism.

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(config.DEVICE)
            angles = angles.to(config.DEVICE)

            outputs = net(images, angles)
            predictions.extend(outputs.cpu().numpy().flatten())

    predictions = np.array(predictions)

    # Get corresponding IDs (sliced by DEBUG_MAX_SAMPLES inside get_dataloaders logic implicitly?
    # Actually data.get_dataloaders slices X_test/inc_test but we need to slice ids_test manually to match)
    ids_test = data_dict["ids_test"]
    if config.DEBUG_MAX_SAMPLES:
        ids_test = ids_test[: config.DEBUG_MAX_SAMPLES]

    # Ensure lengths match
    assert len(predictions) == len(
        ids_test
    ), f"Prediction count {len(predictions)} does not match ID count {len(ids_test)}"

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": predictions})

    # Save Submission
    submission_path = "./demo_submission.csv"
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(df_sub.head())

    # VALIDATION: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created"
    df_check = pd.read_csv(submission_path)
    assert list(df_check.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect submission columns"
    assert len(df_check) == len(ids_test), "Incorrect number of rows in submission"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
