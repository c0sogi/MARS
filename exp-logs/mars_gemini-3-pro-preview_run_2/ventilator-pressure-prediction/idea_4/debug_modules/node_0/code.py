import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import ResBiLSTM
from library.trainer import Trainer

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/demo_execution"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Hyperparameters optimized for a quick demo run
BATCH_SIZE = 256
HIDDEN_DIM = 64  # Reduced from default 512 for speed
NUM_LAYERS = 2  # Reduced from default 4 for speed
EPOCHS = 1  # Single epoch for demonstration
INPUT_DIM = 14  # Based on the 14 engineered features in dataset.py


def main():
    print("=== Starting Ventilator Pressure Prediction Demo ===")

    # 1. Setup and Seeding
    print("\n[Step 1] Setting up environment...")
    seed_everything(42)

    # Clean up previous demo runs if they exist to ensure a fresh start
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading & Feature Engineering
    print("\n[Step 2] Loading data and engineering features...")
    # This function handles loading CSVs, engineering features, splitting based on metadata,
    # reshaping to sequences, and scaling. It caches the result to disk.
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=INPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,  # Force processing from scratch for the demo
        cache_dir=CACHE_DIR,
    )

    # 3. Data Verification
    print("\n[Step 3] Verifying data structures...")
    # Fetch one batch to inspect
    batch = next(iter(train_loader))
    X_batch = batch["X"]
    y_batch = batch["y"]
    u_out_batch = batch["u_out"]

    print(f"Feature Batch Shape: {X_batch.shape}")  # Expected: (Batch, 80, 14)
    print(f"Target Batch Shape: {y_batch.shape}")  # Expected: (Batch, 80)

    # Assertions to ensure data pipeline is correct
    assert X_batch.dim() == 3, "Input X must be 3-dimensional (Batch, Seq, Feat)"
    assert X_batch.shape[1] == 80, "Sequence length must be 80"
    assert X_batch.shape[2] == INPUT_DIM, f"Input dimension must be {INPUT_DIM}"
    assert y_batch.dim() == 2, "Target y must be 2-dimensional (Batch, Seq)"
    assert u_out_batch.shape == y_batch.shape, "u_out mask must match target shape"

    print("Data verification passed.")

    # 4. Model Initialization
    print("\n[Step 4] Initializing ResBiLSTM model...")
    model = ResBiLSTM(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1
    ).to(device)

    # Quick forward pass check
    with torch.no_grad():
        dummy_out = model(X_batch.to(device))
    print(f"Model Output Shape: {dummy_out.shape}")

    assert dummy_out.shape == (BATCH_SIZE, 80, 1), "Model output shape mismatch"
    print("Model initialization passed.")

    # 5. Training
    print("\n[Step 5] Starting Training (Demo: 1 Epoch)...")

    # Initialize Trainer with demo hyperparameters
    trainer = Trainer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        learning_rate=1e-3,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    # We override the internal model of the trainer to ensure it matches our demo config if needed,
    # but the Trainer init creates its own model based on args.
    # The Trainer handles the training loop, validation, and saving the best model.
    trainer.fit(data_dir=INPUT_DIR, cache_dir=CACHE_DIR)

    # Verify model checkpoint was created
    checkpoint_path = os.path.join(CACHE_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Training complete. Best model saved to {checkpoint_path}")
    else:
        raise FileNotFoundError("Model checkpoint not found after training.")

    # 6. Inference / Prediction
    print("\n[Step 6] Generating predictions on Test Set...")

    # The predict method loads the best model state and generates submission.csv
    # It expects the cache_dir to load the processed test data.
    trainer.predict(data_dir=INPUT_DIR, cache_dir=CACHE_DIR)

    # 7. Submission Validation
    print("\n[Step 7] Validating submission file...")
    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission head:\n{df_sub.head()}")
    print(f"Submission shape: {df_sub.shape}")

    # Load test.csv to verify row counts
    df_test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    assert df_sub.shape[1] == 2, "Submission must have 2 columns"
    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Missing required columns"
    assert len(df_sub) == len(
        df_test
    ), f"Row count mismatch: Expected {len(df_test)}, got {len(df_sub)}"

    print("Submission validation passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
