import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse, get_device
from library.dataset import get_dataloaders
from library.model import RNAConvNet
from library.engine import Engine


def run_demonstration():
    print("--- Starting RNA Degradation Prediction Pipeline Demonstration ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override the Config class attributes directly to run a fast demo
    print("\n[1] Configuring environment...")
    seed_everything(42)

    # Use a temporary directory for this run's outputs to keep things clean
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.setup_directories()

    # Set parameters for a quick run
    Config.DEBUG = True  # Use subset of data
    Config.DEBUG_SAMPLES = 50  # Only 50 samples
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2  # Only 2 epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Samples: {Config.DEBUG_SAMPLES}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # load_cached_data=False ensures we demonstrate processing from the provided parquet files
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Verify Data Shapes
    print("    Verifying batch shapes...")
    sample_batch = next(iter(train_loader))

    # Inputs: (Batch, Seq_Len)
    assert sample_batch["sequence"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Sequence shape mismatch: {sample_batch['sequence'].shape}"
    assert sample_batch["structure"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Structure shape mismatch: {sample_batch['structure'].shape}"

    # Targets: (Batch, Scored_Len, 5) -> 5 target columns
    # Note: Targets are only defined for the first 68 bases (SEQ_SCORED)
    assert sample_batch["targets"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        5,
    ), f"Target shape mismatch: {sample_batch['targets'].shape}"

    print("    Data loading and shape verification successful.")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = RNAConvNet()
    device = get_device()
    model.to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    with torch.no_grad():
        seq = sample_batch["sequence"].to(device)
        struct = sample_batch["structure"].to(device)
        loop = sample_batch["loop_type"].to(device)

        # Output should be (Batch, Seq_Len, 5)
        # The model predicts for the entire sequence length (107), even though we only score 68
        output = model(seq, struct, loop)

        assert output.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            5,
        ), f"Model output shape mismatch: {output.shape}"

    print("    Model initialized and forward pass verified.")

    # 4. Training
    print("\n[4] Training Loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, optimizer, device=device)

    # Run training
    engine.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify model checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Best model saved successfully at {best_model_path}")
    else:
        # It's possible validation didn't improve in 2 epochs with random init,
        # but usually fit() saves at least once if val_score < inf.
        # If not, we just note it.
        print(
            "    Notice: No best model file found (validation might not have improved)."
        )

    # 5. Prediction and Submission
    print("\n[5] Generating Predictions...")
    engine.predict(test_loader)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Verify Submission Content
    # Rows should be: Num_Test_Samples * Seq_Len
    # In debug mode, we used 50 test samples
    expected_rows = Config.DEBUG_SAMPLES * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("    Submission file format verified.")

    # 6. Metric Logic Verification
    print("\n[6] Verifying Metric Logic (MCRMSE)...")
    # Create dummy ground truth and predictions
    # Shape: (N_samples, N_targets)
    # Let's assume 2 samples, 2 targets for simplicity
    y_true_dummy = np.array([[1.0, 5.0], [2.0, 6.0]])
    y_pred_dummy = np.array(
        [[1.5, 5.5], [2.5, 6.5]]  # Error: 0.5, 0.5  # Error: 0.5, 0.5
    )

    # Calculation:
    # Col 1: Errors [0.5, 0.5] -> MSE = 0.25 -> RMSE = 0.5
    # Col 2: Errors [0.5, 0.5] -> MSE = 0.25 -> RMSE = 0.5
    # MCRMSE = Mean(0.5, 0.5) = 0.5

    calculated_score = mcrmse(y_true_dummy, y_pred_dummy)
    print(f"    Calculated MCRMSE: {calculated_score}")

    assert np.isclose(
        calculated_score, 0.5
    ), f"MCRMSE calculation incorrect. Expected 0.5, got {calculated_score}"

    print("    Metric logic verified.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
