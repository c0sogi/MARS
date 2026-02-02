import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.train as train_lib


def main():
    print("=== Starting Library Demonstration ===")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Clean working directory to ensure we generate fresh cache with subsampled data
    working_dir = config.WORKING_DIR
    if os.path.exists(working_dir):
        print(f"Cleaning working directory: {working_dir}")
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Override library constants for speed
    # We modify the module attributes directly so functions using them see the changes
    data.DEBUG_SAMPLE_SIZE = 2048  # Small subset for fast processing
    train_lib.EPOCHS = 1  # Only 1 epoch
    train_lib.PATIENCE = 1  # Minimal patience

    # Set seeds
    utils.seed_everything(seed=42)
    device = utils.get_device()
    print(f"Device selected: {device}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Data Pipeline...")

    # We call get_dataloaders. Since we cleared the cache and set DEBUG_SAMPLE_SIZE,
    # this will process a small subset of the data from scratch.
    # Note: run_training calls this internally with load_cached_data=True.
    # We call it here with False first to verify the logic explicitly.
    train_loader, val_loader, test_loader, input_info = data.get_dataloaders(
        load_cached_data=False
    )

    print("DataLoaders created successfully.")
    print(f"Input Info: {input_info}")

    # Assertions for Data
    assert input_info["num_classes"] == 7, "Expected 7 classes (Cover_Type 1-7)"
    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch to verify shapes
    X_cont, X_bin, y = next(iter(train_loader))
    print(f"Batch Shapes - X_cont: {X_cont.shape}, X_bin: {X_bin.shape}, y: {y.shape}")

    assert (
        X_cont.shape[1] == input_info["cont_dim"]
    ), "Continuous feature dimension mismatch"
    assert X_bin.shape[1] == input_info["bin_dim"], "Binary feature dimension mismatch"
    assert (
        y.shape[0] == config.BATCH_SIZE or y.shape[0] == data.DEBUG_SAMPLE_SIZE
    ), f"Batch size mismatch. Expected <= {config.BATCH_SIZE}, got {y.shape[0]}"

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    model = model_lib.ParallelDCNResNet(input_info).to(device)
    model.eval()

    # Create dummy input based on batch info
    dummy_cont = torch.randn(10, input_info["cont_dim"]).to(device)
    dummy_bin = torch.randn(10, input_info["bin_dim"]).to(device)

    with torch.no_grad():
        output = model(dummy_cont, dummy_bin)

    print(f"Model Output Shape: {output.shape}")

    # Assertions for Model
    assert output.shape == (10, 7), f"Expected output shape (10, 7), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ---------------------------------------------------------
    # 4. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    # run_training will re-load data. Since we generated cache in Step 2 (get_dataloaders saves cache),
    # run_training will load the subsampled cached data quickly.
    trained_model, test_loader_from_train = train_lib.run_training()

    assert trained_model is not None, "Training function returned None for model"
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), f"Model file not found at {config.MODEL_SAVE_PATH}"

    # ---------------------------------------------------------
    # 5. Submission Logic Verification
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    trained_model.eval()
    predictions = []
    test_ids_list = []

    # Iterate through test loader to generate predictions
    with torch.no_grad():
        for X_cont, X_bin, ids in test_loader_from_train:
            X_cont = X_cont.to(device)
            X_bin = X_bin.to(device)

            outputs = trained_model(X_cont, X_bin)
            _, preds = torch.max(outputs, 1)

            # Map 0-6 back to 1-7 (Cover_Type)
            preds = preds + 1

            predictions.extend(preds.cpu().numpy())
            test_ids_list.extend(ids.numpy())

    # Save submission
    utils.save_submission(predictions, test_ids_list, save_path=config.SUBMISSION_FILE)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Submission columns missing"
    assert df_sub.shape[0] == len(test_ids_list), "Submission row count mismatch"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
