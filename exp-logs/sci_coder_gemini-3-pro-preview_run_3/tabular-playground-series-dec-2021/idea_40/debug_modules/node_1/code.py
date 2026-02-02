import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.data_utils import get_dataloaders
from library.model_utils import DeepSupervisedHybridModel
from library.train_utils import run_training, predict, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Demo Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration run
    print("Configuring parameters for demo run...")
    Config.EPOCHS = 1  # Reduce epochs for speed
    Config.BATCH_SIZE = 8192  # Increase batch size for faster iteration on A100
    Config.WORKING_DIR = "./working/demo_execution"  # Separate working dir
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("\n[Step 1] Loading and Processing Data...")
    # We force re-processing (load_cached_data=False) to demonstrate the full pipeline
    # In a real scenario, set this to True to use cached .npy files
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verification: Check DataLoaders
    print("Verifying DataLoader integrity...")
    try:
        train_batch, train_labels = next(iter(train_loader))
        val_batch, val_labels = next(iter(val_loader))
        test_batch = next(iter(test_loader))
    except StopIteration:
        raise AssertionError("DataLoaders are empty!")

    # Check shapes
    input_dim = train_batch.shape[1]
    print(f"  Input feature dimension: {input_dim}")
    print(f"  Batch size: {train_batch.shape[0]}")

    assert train_batch.dim() == 2, "Train input should be 2D [batch, features]"
    assert train_labels.dim() == 1, "Train labels should be 1D [batch]"
    assert (
        train_batch.shape[1] == test_batch.shape[1]
    ), "Train and Test feature dimensions mismatch"

    # Check Label Range (should be 0-6 internally)
    assert (
        train_labels.min() >= 0 and train_labels.max() < Config.NUM_CLASSES
    ), f"Labels out of range [0, {Config.NUM_CLASSES-1}]"

    print("Data verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[Step 2] Verifying Model Architecture...")
    device = Config.DEVICE

    # Instantiate model manually to check forward pass logic
    model = DeepSupervisedHybridModel(
        input_dim=input_dim, num_classes=Config.NUM_CLASSES, hidden_dim=512
    ).to(device)

    # Create dummy input
    dummy_input = torch.randn(32, input_dim).to(device)

    # Forward pass
    prim_logits, aux_logits = model(dummy_input)

    # Assertions
    assert prim_logits.shape == (
        32,
        Config.NUM_CLASSES,
    ), f"Primary logits shape mismatch: expected (32, {Config.NUM_CLASSES}), got {prim_logits.shape}"
    assert aux_logits.shape == (
        32,
        Config.NUM_CLASSES,
    ), f"Auxiliary logits shape mismatch: expected (32, {Config.NUM_CLASSES}), got {aux_logits.shape}"

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[Step 3] Running Training Loop...")
    # run_training handles model instantiation, training, validation, and saving best model
    trained_model = run_training(train_loader, val_loader)

    # Verify model artifact creation
    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to save model to {expected_model_path}"
        )

    print(f"Training complete. Model saved to {expected_model_path}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 4] Generating Predictions...")
    # predict handles inference and CSV generation
    predict(trained_model, test_loader, test_ids)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Prediction failed to save submission to {Config.SUBMISSION_FILE}"
        )

    # Load submission to verify format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Loaded submission file with shape: {df_sub.shape}")

    # Assertions on submission
    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch: {len(df_sub)} vs {len(test_ids)}"
    assert df_sub["Cover_Type"].isnull().sum() == 0, "Submission contains NaN values"

    # Check if predictions are in original 1-7 range (predict function adds 1)
    # Note: If model is untrained/random, it might predict any class, but range must be valid.
    # The dataset has classes 1-7.
    unique_preds = df_sub["Cover_Type"].unique()
    print(f"Unique predicted classes: {sorted(unique_preds)}")
    assert df_sub["Cover_Type"].min() >= 1, "Predicted class < 1 found"
    assert df_sub["Cover_Type"].max() <= 7, "Predicted class > 7 found"

    print("Submission verification passed.")
    print("\nDemo script executed successfully.")


if __name__ == "__main__":
    main()
