import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.dataset import prepare_data
from library.model import LGRHNet
from library.train import run_training
from library.utils import seed_everything, get_device


def main():
    print("Initializing Demonstration Script...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Configuration Setup
    # We override defaults to ensure the script runs quickly (Debug Mode)
    print("Setting up configuration...")
    config = Config(
        debug=True,  # Use subset of data (100 train, 50 val breaths)
        working_dir=os.path.dirname(demo_dir),
        cache_dir=demo_dir,
        submission_dir=demo_dir,
        submission_file=os.path.join(demo_dir, "submission.csv"),
        model_save_path=os.path.join(demo_dir, "best_model.pth"),
        scaler_save_path=os.path.join(demo_dir, "scaler.joblib"),
        # Reduced Model Complexity for Speed
        tcn_layers=2,
        tcn_filters=32,
        lstm_layers=1,
        lstm_hidden_size=64,
        lstm_bidirectional=True,
        fusion_hidden_size=64,
        # Training Hyperparameters
        epochs=2,
        batch_size=16,  # Small batch size for debug data
        num_workers=0,  # Avoid multiprocessing overhead in demo
        lookahead_steps=2,  # Reduced feature engineering cost
    )

    # Set seed for reproducibility
    seed_everything(config.seed)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Pipeline Verification
    print("\n=== Verifying Data Pipeline ===")
    # We manually call prepare_data to inspect the output before training
    train_loader, val_loader, test_loader = prepare_data(config)

    # Assertions to ensure data is loaded correctly
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Inspect a single batch
    X_batch, y_batch, u_out_batch = next(iter(train_loader))

    print(
        f"Batch Shapes - X: {X_batch.shape}, y: {y_batch.shape}, u_out: {u_out_batch.shape}"
    )

    # Expected shapes: (Batch, Seq_Len=80, Features)
    # Note: Seq_Len is fixed at 80 for this dataset
    assert X_batch.ndim == 3, "Input X should be 3-dimensional (Batch, Seq, Feat)"
    assert y_batch.ndim == 2, "Target y should be 2-dimensional (Batch, Seq)"
    assert u_out_batch.ndim == 2, "Mask u_out should be 2-dimensional (Batch, Seq)"
    assert X_batch.shape[1] == 80, "Sequence length must be 80"

    input_dim = X_batch.shape[2]
    print(f"Input Feature Dimension: {input_dim}")

    # 3. Model Architecture Verification
    print("\n=== Verifying Model Architecture ===")
    model = LGRHNet(input_dim=input_dim, config=config)
    model.to(device)

    # Perform a dummy forward pass
    # Move batch to device
    X_device = X_batch.to(device)

    with torch.no_grad():
        output = model(X_device)

    print(f"Model Output Shape: {output.shape}")

    # Expected output: (Batch, Seq_Len, 1)
    assert output.shape == (
        X_batch.shape[0],
        80,
        1,
    ), f"Expected output shape {(X_batch.shape[0], 80, 1)}, got {output.shape}"

    print("Model forward pass successful.")

    # 4. Training Loop Execution
    print("\n=== Executing Training Loop (Debug Mode) ===")
    # run_training handles the full loop: train, validate, save model, predict
    # It will re-use the cached data we just generated in step 2
    run_training(config)

    # 5. Output Verification
    print("\n=== Verifying Outputs ===")

    # Check if model file exists
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(f"Model file not found at {config.model_save_path}")
    print("Model file generated successfully.")

    # Check if submission file exists
    if not os.path.exists(config.submission_file):
        raise FileNotFoundError(
            f"Submission file not found at {config.submission_file}"
        )

    # Validate submission format
    sub_df = pd.read_csv(config.submission_file)
    print(f"Submission Head:\n{sub_df.head()}")

    required_cols = {"id", "pressure"}
    if not required_cols.issubset(sub_df.columns):
        raise ValueError(
            f"Submission missing required columns. Found: {sub_df.columns}"
        )

    if sub_df.isnull().any().any():
        raise ValueError("Submission contains NaN values.")

    print(f"Submission generated with {len(sub_df)} rows.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
