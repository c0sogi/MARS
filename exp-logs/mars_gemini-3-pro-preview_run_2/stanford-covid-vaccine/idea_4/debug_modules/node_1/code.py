import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
from library import config, utils, data, model, train, predict


def main():
    print("=== Starting RNA Degradation Prediction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Override config for speed and isolation
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    config.NUM_EPOCHS = 2  # Train for only 2 epochs
    config.BATCH_SIZE = 16  # Smaller batch size for the small subset
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "data_cache")
    config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Create directories
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR)
    os.makedirs(config.CACHE_DIR)

    # Set seeds
    utils.seed_everything(config.SEED)
    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Debug Mode: {config.DEBUG}")
    print(f"    Epochs: {config.NUM_EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Force processing from CSVs (load_cached_data=False) to test preprocessing logic
    # This will also save cache for the subsequent training step
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=False)

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"    Input Batch Shape: {inputs.shape}")
    print(f"    Target Batch Shape: {targets.shape}")

    # Assertions
    # Inputs: (Batch, Seq_Len=107, Channels=18)
    assert (
        inputs.shape[1] == 107
    ), f"Expected sequence length 107, got {inputs.shape[1]}"
    assert inputs.shape[2] == 18, f"Expected 18 input channels, got {inputs.shape[2]}"

    # Targets: (Batch, Seq_Scored=68, Targets=5)
    assert (
        targets.shape[1] == 68
    ), f"Expected scored sequence length 68, got {targets.shape[1]}"
    assert targets.shape[2] == 5, f"Expected 5 target columns, got {targets.shape[2]}"

    print("    Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape verification
    net = model.PartnerAwareHybridNet().to(device)

    # Forward pass
    outputs = net(inputs.to(device))
    print(f"    Output Batch Shape: {outputs.shape}")

    # Assertions
    # Outputs: (Batch, Seq_Len=107, Targets=5)
    # Note: The model outputs predictions for the full 107 length, loss handles slicing
    assert (
        outputs.shape[1] == 107
    ), f"Expected output sequence length 107, got {outputs.shape[1]}"
    assert outputs.shape[2] == 5, f"Expected 5 output targets, got {outputs.shape[2]}"

    # Loss Calculation Check
    criterion = utils.MCRMSELoss()
    loss = criterion(outputs, targets.to(device))
    print(f"    Initial Loss (MCRMSE): {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    print("    Model forward pass and loss calculation verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Reduced Epochs)...")

    # This function uses the config values we set earlier
    train.run_training()

    # Verify model file creation
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file was not saved."
    file_size = os.path.getsize(config.MODEL_SAVE_PATH)
    print(
        f"    Training complete. Model saved to {config.MODEL_SAVE_PATH} ({file_size/1024:.2f} KB)"
    )

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    predict.generate_submission()

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Assertions on Submission
    # Rows should be: N_test_samples * Seq_Length.
    # In debug mode, we have 50 test samples. 50 * 107 = 5350 rows.
    expected_rows = config.DEBUG_SUBSET_SIZE * config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check required columns
    required_cols = ["id_seqpos"] + config.TARGET_COLS
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission."

    # Check content format
    sample_id_pos = df_sub.iloc[0]["id_seqpos"]
    assert "id_" in sample_id_pos, "id_seqpos format seems incorrect."

    print("    Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
