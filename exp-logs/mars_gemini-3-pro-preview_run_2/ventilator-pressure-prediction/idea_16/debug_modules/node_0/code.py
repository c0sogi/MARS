import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HFSI_BiLSTM
from library.train import train_model
from library.predict import generate_submission
from library.utils import seed_everything, compute_metric, get_device

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline_demo():
    print("=== Starting Ventilator Pressure Prediction Pipeline Demo ===")

    # 1. Configuration Setup
    # Initialize Config with debug=True to reduce epochs (2) and batch size (64)
    config = Config(debug=True)

    # Override paths to use the provided 'demo_data' for speed optimization.
    # This ensures we don't process the full 5GB dataset for this demonstration.
    demo_data_dir = "./working/demo_data"
    config.TRAIN_CSV = os.path.join(demo_data_dir, "mini_train.csv")
    config.TEST_CSV = os.path.join(demo_data_dir, "mini_test.csv")
    config.TRAIN_META = os.path.join(demo_data_dir, "train_meta.csv")
    config.VAL_META = os.path.join(demo_data_dir, "val_meta.csv")
    config.TEST_META = os.path.join(demo_data_dir, "test_meta.csv")

    # Set a specific working directory for this demo to ensure a clean state
    demo_working_dir = "./working/submission_demo"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    # Update cache and output paths to the new working directory
    config.WORKING_DIR = demo_working_dir
    config.TRAIN_CACHE = os.path.join(demo_working_dir, "train_processed.parquet")
    config.VAL_CACHE = os.path.join(demo_working_dir, "val_processed.parquet")
    config.TEST_CACHE = os.path.join(demo_working_dir, "test_processed.parquet")
    config.SCALER_CACHE = os.path.join(demo_working_dir, "scaler_params.npy")
    config.MODEL_CHECKPOINT = os.path.join(demo_working_dir, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Ensure reproducibility
    seed_everything(config.SEED)

    print(f"Configuration set. Output directory: {config.WORKING_DIR}")

    # 2. Data Loading & Verification
    print("\n[Step 1] Loading and Verifying Data...")
    # This handles feature engineering, scaling, and reshaping
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(config)

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    inputs = batch["input"]
    targets = batch["target"]
    u_out = batch["u_out"]

    # Assertions for data integrity
    # Expected shape: (Batch_Size, Seq_Len=80, Num_Features)
    assert inputs.ndim == 3, f"Input tensor must be 3D, got {inputs.shape}"
    assert inputs.shape[1] == 80, f"Sequence length must be 80, got {inputs.shape[1]}"
    assert targets.shape == (
        config.BATCH_SIZE,
        80,
    ), f"Target shape mismatch: {targets.shape}"
    assert u_out.shape == (
        config.BATCH_SIZE,
        80,
    ), f"u_out shape mismatch: {u_out.shape}"

    input_dim = inputs.shape[-1]
    print(f"Data verification passed. Input Feature Dimension: {input_dim}")

    # 3. Model Architecture Verification
    print("\n[Step 2] Verifying Model Architecture...")
    device = get_device()
    model = HFSI_BiLSTM(config, input_dim).to(device)

    # Perform a dummy forward pass
    dummy_input = inputs.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (Batch_Size, Seq_Len=80, 1)
    assert output.shape == (
        config.BATCH_SIZE,
        80,
        1,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass successful.")

    # 4. Metric Logic Verification
    print("\n[Step 3] Verifying Metric Calculation (Inspiratory Phase Only)...")
    # Synthetic Data:
    # 2 time steps:
    #   Step 1: u_out=0 (Inspiratory), Pred=10, Target=10 (Error=0)
    #   Step 2: u_out=1 (Expiratory), Pred=100, Target=10 (Error=90)
    # The metric should ignore Step 2, resulting in MAE = 0.0
    synth_preds = np.array([10, 100])
    synth_targets = np.array([10, 10])
    synth_u_out = np.array([0, 1])

    mae = compute_metric(synth_preds, synth_targets, synth_u_out)
    assert (
        mae == 0.0
    ), f"Metric failed to ignore expiratory phase. Expected 0.0, got {mae}"
    print("Metric calculation verified.")

    # 5. Training Loop Execution
    print("\n[Step 4] Executing Training Loop...")
    # Runs for 2 epochs on the mini dataset (debug mode)
    train_model(config)

    # Verify checkpoint creation
    if not os.path.exists(config.MODEL_CHECKPOINT):
        raise FileNotFoundError("Model checkpoint was not saved after training.")
    print("Training complete. Best model saved.")

    # 6. Inference & Submission
    print("\n[Step 5] Generating Submission...")
    generate_submission(config)

    # Verify submission file
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Verify columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "pressure" in df_sub.columns, "Submission missing 'pressure' column"

    # Verify row count matches the test IDs processed
    # Note: In debug mode, data is subsampled to 100 breaths * 80 steps = 8000 rows
    expected_rows = test_ids.size
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print(f"Submission verified. Shape: {df_sub.shape}")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
