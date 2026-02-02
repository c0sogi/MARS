import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, masked_mae_metric
from library.dataset import get_data_loaders
from library.model import VentilatorModel
from library.train import run_training
from library.predict import generate_submission


def main():
    print("=== Ventilator Pressure Prediction Pipeline Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up configuration for fast demonstration...")
    seed_everything(Config.SEED)

    # Modify Config for a quick run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Define a debug limit (number of breaths)
    # 16 batch size * 5 batches = 80 samples
    DEBUG_LIMIT = 80

    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Limit: {DEBUG_LIMIT} breaths")

    # 2. Data Loading & Verification
    print("\n[2] Verifying Data Loading and Preprocessing...")

    # We force reload_cached_data=False to demonstrate the processing pipeline at least once,
    # or rely on the library's caching mechanism. Since we want to test the pipeline,
    # we'll let the library handle it.
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug_limit=DEBUG_LIMIT,
    )

    # Fetch one batch to verify shapes
    X_batch, y_batch, u_out_batch = next(iter(train_loader))

    print(f"    Train Batch X shape: {X_batch.shape}")
    print(f"    Train Batch y shape: {y_batch.shape}")
    print(f"    Train Batch u_out shape: {u_out_batch.shape}")

    # Assertions
    # Shape: (Batch, Seq_Len, Features)
    assert X_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_FEATURES,
    ), f"Incorrect X shape. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_FEATURES)}, got {X_batch.shape}"

    # Shape: (Batch, Seq_Len)
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect y shape. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {y_batch.shape}"

    assert u_out_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect u_out shape. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {u_out_batch.shape}"

    print("    Data shapes verified successfully.")

    # 3. Metric Logic Verification
    print("\n[3] Verifying Metric Logic (Masked MAE)...")
    # Create synthetic data
    # Case: 2 time steps.
    # Step 0: u_out=0 (Inspiratory, should count). True=10, Pred=12. Error=2.
    # Step 1: u_out=1 (Expiratory, should NOT count). True=10, Pred=100. Error=90 (Ignored).

    y_true_synth = torch.tensor([[10.0, 10.0]])
    y_pred_synth = torch.tensor([[12.0, 100.0]])
    u_out_synth = torch.tensor([[0.0, 1.0]])

    mae = masked_mae_metric(y_pred_synth, y_true_synth, u_out_synth)
    expected_mae = 2.0 / 1.0  # Total error 2 / 1 valid step

    print(f"    Calculated MAE: {mae}, Expected: {expected_mae}")
    assert abs(mae - expected_mae) < 1e-6, "Masked MAE calculation is incorrect!"
    print("    Metric logic verified successfully.")

    # 4. Model Instantiation & Forward Pass
    print("\n[4] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = VentilatorModel(config=Config).to(device)

    # Move batch to device
    X_batch = X_batch.to(device)

    # Forward pass
    final_pred, aux_pred = model(X_batch)

    print(f"    Model Output Shape (Final): {final_pred.shape}")
    if aux_pred is not None:
        print(f"    Model Output Shape (Aux): {aux_pred.shape}")

    assert final_pred.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Model final output shape mismatch."

    if aux_pred is not None:
        assert aux_pred.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
        ), "Model aux output shape mismatch."

    print("    Model forward pass verified successfully.")

    # 5. Training Loop Execution
    print("\n[5] Running Training Loop (Demo)...")
    demo_model_name = "demo_model.pth"

    # Clean up any previous demo model
    model_path = os.path.join(Config.WORKING_DIR, demo_model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    run_training(
        debug_limit=DEBUG_LIMIT, load_cached_data=True, save_path=demo_model_name
    )

    # Verify model file creation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Training failed to save model at {model_path}")

    print(f"    Model successfully saved to {model_path}")

    # 6. Inference & Submission
    print("\n[6] Running Inference and Generating Submission...")

    # Clean up previous submission
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    generate_submission(
        model_filename=demo_model_name, load_cached_data=True, debug_limit=DEBUG_LIMIT
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to save submission at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Submission columns: {list(sub_df.columns)}")

    # Check shape: debug_limit * 80 rows
    expected_rows = DEBUG_LIMIT * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    assert (
        Config.ID in sub_df.columns and Config.PRESSURE in sub_df.columns
    ), "Submission columns mismatch."

    print("    Submission generated and verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
