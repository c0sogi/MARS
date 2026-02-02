import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import random

# Import from the provided library files
from library.config import Config
from library.data_loader import get_data_loaders
from library.model import HybridLSTMTransformer
from library.trainer import Trainer
from library.inference import run_inference


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed}")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Configuring Environment for Demonstration ---")
    set_seed(42)

    # Override Config for a fast, isolated demo run
    # We create a specific directory for this execution to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths dynamically
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_data.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_data.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_data.npz")
    Config.SCALER_CACHE = os.path.join(demo_dir, "scaler_params.npz")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Update Hyperparameters for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.DEBUG = True  # Use debug mode (subset of data)
    Config.DEBUG_SAMPLES = 200  # Use 200 breaths for train/val/test
    Config.BATCH_SIZE = 32  # Small batch size

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n--- 2. Verifying Data Loading & Processing ---")

    # Force processing from scratch (load_cached_data=False) to test feature engineering
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=False, debug=True
    )

    # Verify Train Loader
    print("Checking Train Loader...")
    train_batch = next(iter(train_loader))

    # Check keys
    assert "cont" in train_batch, "Missing continuous features in batch"
    assert "cat" in train_batch, "Missing categorical features in batch"
    assert "target" in train_batch, "Missing targets in batch"

    # Check Shapes
    # Expected: (Batch, Seq_Len, Features)
    # Seq_Len is 80 (Config.SEQ_LEN)
    # Cont Features is 10 (Config.NUM_CONT_FEATURES)
    # Cat Features is 2 (R and C)
    B, S, F_cont = train_batch["cont"].shape
    _, _, F_cat = train_batch["cat"].shape

    assert (
        B == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {B}"
    assert (
        S == Config.SEQ_LEN
    ), f"Sequence length mismatch. Expected {Config.SEQ_LEN}, got {S}"
    assert (
        F_cont == Config.NUM_CONT_FEATURES
    ), f"Cont feature count mismatch. Expected {Config.NUM_CONT_FEATURES}, got {F_cont}"
    assert F_cat == 2, f"Cat feature count mismatch. Expected 2, got {F_cat}"

    print("Data Loader shapes verified successfully.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n--- 3. Verifying Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = HybridLSTMTransformer().to(device)

    # Move batch to device
    cont_x = train_batch["cont"].to(device)
    cat_x = train_batch["cat"].to(device)

    # Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        preds = model(cont_x, cat_x)

    # Check Output Shape: (Batch, Seq_Len)
    assert preds.shape == (
        B,
        S,
    ), f"Model output shape mismatch. Expected {(B, S)}, got {preds.shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Verification
    # ==========================================
    print("\n--- 4. Verifying Training Loop ---")

    # Initialize Trainer
    # We use load_cached_data=True because we generated the cache in step 2
    trainer = Trainer(load_cached_data=True, debug=True)

    # Run training
    print("Starting training (1 epoch)...")
    trainer.fit(epochs=Config.EPOCHS)

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
    print(f"Model saved successfully to {Config.BEST_MODEL_PATH}")

    # ==========================================
    # 5. Inference Pipeline Verification
    # ==========================================
    print("\n--- 5. Verifying Inference Pipeline ---")

    # Run Inference
    # This will load the best model saved in step 4 and predict on the test set
    run_inference(load_cached_data=True, debug=True)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded submission file. Shape: {df_sub.shape}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "pressure" in df_sub.columns, "Submission missing 'pressure' column"

    # Check for empty submission
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if pressure values are numeric
    assert pd.api.types.is_numeric_dtype(
        df_sub["pressure"]
    ), "Pressure predictions are not numeric"

    print("Inference pipeline completed successfully.")
    print("\n==========================================")
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
