import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library
from library.utils import seed_everything, get_device
from library.dataset import prepare_data_loaders
from library.model import DPI_BiLSTM
from library.loss_metric import WeightedL1Loss
from library.engine import fit, predict_and_submit
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    PROJECTION_DIM,
    NUM_LSTM_LAYERS,
    SUBMISSION_DIR,
)


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting up environment...")
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Preparation (Debug Mode)
    # We use debug=True to load only a small subset (100 breaths) for speed.
    print("\n[Step 2] Preparing DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        batch_size=16,
        num_workers=2,
        debug=True,
        load_cached_data=False,  # Force processing to demonstrate the pipeline
    )

    # Verify Data Shapes
    print("Verifying data shapes...")
    sample_batch = next(iter(train_loader))
    x_batch = sample_batch["x"]
    y_batch = sample_batch["y"]
    u_out_batch = sample_batch["u_out"]

    # Expected shapes: (Batch, Seq_Len, Features) and (Batch, Seq_Len)
    # Seq_Len is fixed at 80 in config.
    expected_seq_len = 80

    assert x_batch.dim() == 3, f"Input X must be 3D, got {x_batch.shape}"
    assert (
        x_batch.shape[1] == expected_seq_len
    ), f"Seq len must be {expected_seq_len}, got {x_batch.shape[1]}"
    assert (
        x_batch.shape[2] == INPUT_DIM
    ), f"Feature dim must be {INPUT_DIM}, got {x_batch.shape[2]}"
    assert y_batch.shape == (
        16,
        expected_seq_len,
    ), f"Target Y shape mismatch: {y_batch.shape}"
    assert u_out_batch.shape == (
        16,
        expected_seq_len,
    ), f"u_out shape mismatch: {u_out_batch.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization
    print("\n[Step 3] Initializing DPI-BiLSTM Model...")
    model = DPI_BiLSTM(
        input_dim=INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,  # Reduced layers for demo speed
        dropout=0.1,
    )
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = x_batch.to(device)
        dummy_output = model(dummy_input)
        # Output should be (Batch, Seq_Len, 1)
        assert dummy_output.shape == (
            16,
            expected_seq_len,
            1,
        ), f"Model output shape mismatch: {dummy_output.shape}"
    print("Model initialized and forward pass verified.")

    # 4. Training Setup
    print("\n[Step 4] Setting up Training Components...")
    criterion = WeightedL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    # 5. Training Execution
    print("\n[Step 5] Running Training Loop (2 Epochs)...")
    # We use a temporary path for the model checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "working", "demo_model.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=2,
        patience=1,  # Early stopping won't likely trigger in 2 epochs
        device=device,
        save_path=checkpoint_path,
    )
    print("Training loop completed.")

    # 6. Inference and Submission
    print("\n[Step 6] Generating Submission...")
    submission_path = os.path.join(SUBMISSION_DIR, "demo_submission.csv")

    predict_and_submit(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        output_path=submission_path,
    )

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created."
    df_sub = pd.read_csv(submission_path)

    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check for NaNs
    if df_sub.isnull().sum().sum() > 0:
        raise AssertionError("Submission contains NaN values.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
