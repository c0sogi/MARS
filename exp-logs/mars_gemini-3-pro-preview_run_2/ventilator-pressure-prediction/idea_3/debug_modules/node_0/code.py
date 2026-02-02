import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.utils import seed_everything, get_device
from library.dataset import get_data_loaders
from library.model import HybridModel
from library.train import train_model
from library.inference import generate_predictions


def run_demo():
    # 1. Setup and Configuration
    print("=== Starting Demonstration ===")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define a temporary directory for this demo to avoid overwriting existing work
    demo_dir = "./working/demo_execution_test"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Demonstrate Data Loading (Dataset & DataLoader)
    print("\n=== Testing Data Loading (Debug Mode) ===")
    # We use debug=True to load a tiny subset of data for speed
    batch_size = 8
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        load_cached_data=False,  # Force processing to verify logic
        debug=True,
    )

    # Fetch a single batch to verify structure and shapes
    batch = next(iter(train_loader))
    X_batch = batch["input"]
    y_batch = batch["target"]
    u_out_batch = batch["u_out"]

    print(f"Batch Input Shape: {X_batch.shape}")  # Expected: (Batch, 80, 14)
    print(f"Batch Target Shape: {y_batch.shape}")  # Expected: (Batch, 80)
    print(f"Batch u_out Shape: {u_out_batch.shape}")  # Expected: (Batch, 80)

    # Assertions
    # Feature count = 11 continuous + 1 u_out + 2 categorical (R, C) = 14
    assert X_batch.shape == (
        batch_size,
        80,
        14,
    ), f"Expected input shape ({batch_size}, 80, 14), got {X_batch.shape}"
    assert y_batch.shape == (
        batch_size,
        80,
    ), f"Expected target shape ({batch_size}, 80), got {y_batch.shape}"
    assert u_out_batch.shape == (
        batch_size,
        80,
    ), f"Expected u_out shape ({batch_size}, 80), got {u_out_batch.shape}"
    print("Data Loading Verification Passed.")

    # 3. Demonstrate Model Architecture
    print("\n=== Testing Model Architecture ===")
    # Initialize model with default parameters matching the dataset
    # input_dim=12 because the model splits the last 2 cols as categorical, leaving 12 continuous
    model = HybridModel(
        input_dim=12,
        lstm_dim=64,  # Reduced for demo speed
        num_lstm_layers=2,  # Reduced for demo speed
        emb_dim=4,
        cnn_dim=32,  # Reduced for demo speed
    ).to(device)

    # Move batch to device
    X_batch = X_batch.to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(X_batch)

    print(f"Model Output Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        batch_size,
        80,
    ), f"Expected output shape ({batch_size}, 80), got {preds.shape}"
    print("Model Forward Pass Verification Passed.")

    # 4. Demonstrate Training Loop
    print("\n=== Testing Training Loop ===")
    # Run a very short training cycle (1 epoch) on debug data
    train_model(
        epochs=1,
        batch_size=batch_size,
        lr=1e-3,
        debug=True,
        patience=1,
        save_dir=demo_dir,
    )

    # Verify artifacts
    best_model_path = os.path.join(demo_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Training failed to save best_model.pth"
    print(f"Training Verification Passed. Model saved to {best_model_path}")

    # 5. Demonstrate Inference
    print("\n=== Testing Inference Pipeline ===")
    # Generate predictions using the model we just trained
    generate_predictions(
        model_path=best_model_path,
        batch_size=batch_size,
        debug=True,
        submission_output_dir=demo_dir,
        load_cached_data=True,  # Use the cached debug data from step 2/4
    )

    submission_path = os.path.join(demo_dir, "submission.csv")
    assert os.path.exists(submission_path), "Inference failed to save submission.csv"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"
    print("Inference Verification Passed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
