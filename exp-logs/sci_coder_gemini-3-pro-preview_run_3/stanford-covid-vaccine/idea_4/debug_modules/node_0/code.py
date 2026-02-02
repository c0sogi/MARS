import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.data_processor import get_dataloaders, RNADataset
from library.model import RNAConformer
from library.trainer import Trainer
from library.utils import seed_everything, compute_mcrmse, create_submission


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Setup and Configuration Override
    # We override specific configurations to ensure the demo runs quickly.
    print("\n[1] Configuring environment...")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_WORKING_DIR, "train_data.npy")
    Config.VAL_CACHE = os.path.join(DEMO_WORKING_DIR, "val_data.npy")
    Config.TEST_CACHE = os.path.join(DEMO_WORKING_DIR, "test_data.npy")
    Config.FINAL_SUBMISSION = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.PATIENCE = 2

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading and Processing
    print("\n[2] Loading and processing data...")
    # This will read from ./metadata/*.parquet and save cache to DEMO_WORKING_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing for demonstration
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple script execution stability
    )

    # Verify Data Shapes
    print("Verifying data shapes...")
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["inputs"]
    targets = sample_batch["targets"]
    ids = sample_batch["ids"]

    # Expected: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"

    # Expected: (Batch, Seq_Len=107, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.OUTPUT_CHANNELS,
    ), f"Target shape mismatch: {targets.shape}"

    print(f"Train Batch Inputs: {inputs.shape}")
    print(f"Train Batch Targets: {targets.shape}")
    print("Data loading verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Initializing Model...")
    device = Config.DEVICE
    model = RNAConformer().to(device)

    # Test Forward Pass
    dummy_input = inputs.to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.OUTPUT_CHANNELS,
    ), f"Model output shape mismatch: {dummy_output.shape}"

    print(f"Model Output Shape: {dummy_output.shape}")
    print("Model initialization verification passed.")

    # 4. Metric Verification
    print("\n[4] Verifying Metric Logic (MCRMSE)...")
    # Create dummy predictions and targets
    # Case 1: Perfect prediction
    dummy_preds = np.random.rand(10, 68, 5)
    dummy_targets = dummy_preds.copy()
    score_perfect = compute_mcrmse(dummy_preds, dummy_targets)
    assert np.isclose(
        score_perfect, 0.0
    ), f"Perfect score should be 0.0, got {score_perfect}"

    # Case 2: Known error
    # Preds = 1, Targets = 0 -> Error = 1, RMSE = 1, MCRMSE = 1
    dummy_preds_ones = np.ones((10, 68, 5))
    dummy_targets_zeros = np.zeros((10, 68, 5))
    score_ones = compute_mcrmse(dummy_preds_ones, dummy_targets_zeros)
    assert np.isclose(score_ones, 1.0), f"Score should be 1.0, got {score_ones}"

    print("Metric logic verification passed.")

    # 5. Training Loop
    print("\n[5] Starting Training Loop...")
    trainer = Trainer(model=model, device=device)

    # Update trainer's best model path to our working dir
    trainer.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    trainer.fit(train_loader, val_loader)

    assert os.path.exists(
        trainer.best_model_path
    ), "Best model checkpoint was not saved."
    print("Training loop completed successfully.")

    # 6. Inference and Submission
    print("\n[6] Generating Predictions and Submission...")

    # Load best model
    trainer.load_best_model()

    # Predict on test set
    test_preds = trainer.predict(test_loader)

    # Get Test IDs from the dataset directly
    # The DataLoader wraps the dataset, so we access the underlying dataset
    test_ids = test_loader.dataset.ids

    print(f"Predictions shape: {test_preds.shape}")
    print(f"Number of Test IDs: {len(test_ids)}")

    # Validate prediction shape (N_test, 107, 5)
    assert (
        test_preds.shape[1] == Config.SEQ_LENGTH
    ), "Prediction sequence length mismatch."
    assert (
        test_preds.shape[2] == Config.OUTPUT_CHANNELS
    ), "Prediction channel count mismatch."
    assert (
        len(test_ids) == test_preds.shape[0]
    ), "Mismatch between IDs and prediction count."

    # Generate Submission CSV
    submission_df = create_submission(
        test_ids, test_preds, save_path=Config.FINAL_SUBMISSION
    )

    # Verify Submission File
    assert os.path.exists(Config.FINAL_SUBMISSION), "Submission file not found."

    # Check submission content format
    # Expected rows: N_test * 107
    expected_rows = len(test_ids) * Config.SEQ_LENGTH
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {submission_df.columns}"

    print(f"Submission generated at {Config.FINAL_SUBMISSION}")
    print("Head of submission:")
    print(submission_df.head())

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
