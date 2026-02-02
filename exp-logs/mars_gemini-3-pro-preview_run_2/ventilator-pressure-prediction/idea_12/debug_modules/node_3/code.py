import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.dataset import get_dataloaders
from library.model import DP_GI_BiLSTM
from library.train import Trainer
from library.inference import predict


def run_demonstration():
    print("Starting Ventilator Pressure Prediction Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for Debug execution...")

    # Enable Debug Mode for speed (Process fewer breaths, fewer epochs)
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128  # Efficient batch size for A100

    # Setup isolated working directory for this demo
    demo_dir = "./working/demo_execution_script"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths to point to the demo directory
    # Note: Since these are class attributes initialized at import, we must update them manually
    Config.WORKING_DIR = demo_dir
    Config.CACHE_TRAIN_PATH = os.path.join(demo_dir, "train_processed_debug.parquet")
    Config.CACHE_VAL_PATH = os.path.join(demo_dir, "val_processed_debug.parquet")
    Config.CACHE_TEST_PATH = os.path.join(demo_dir, "test_processed_debug.parquet")
    Config.SCALER_PATH = os.path.join(demo_dir, "scaler_params.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Logic Verification: Loss Function
    # ==========================================
    print("\n[2] Verifying WeightedL1Loss logic...")

    # Define a deterministic case
    # Inspiratory Phase (u_out=0): Weight should be 1.0
    # Expiratory Phase (u_out=1): Weight should be 0.1
    criterion = WeightedL1Loss(inspiratory_weight=1.0, expiratory_weight=0.1)

    dummy_pred = torch.tensor([10.0, 10.0])
    dummy_target = torch.tensor([20.0, 20.0])
    dummy_u_out = torch.tensor([0.0, 1.0])  # First is insp, second is exp

    # Calculation:
    # Item 1: |10-20| * 1.0 = 10.0
    # Item 2: |10-20| * 0.1 = 1.0
    # Mean: (10 + 1) / 2 = 5.5

    loss_val = criterion(dummy_pred, dummy_target, dummy_u_out).item()

    assert (
        abs(loss_val - 5.5) < 1e-5
    ), f"Loss verification failed. Expected 5.5, got {loss_val}"
    print("WeightedL1Loss verified successfully.")

    # ==========================================
    # 3. Data Processing & Loading
    # ==========================================
    print("\n[3] Processing data and creating DataLoaders...")

    # Force processing from scratch (load_cached_data=False) to generate debug subsets
    # This will read raw CSVs, sample breaths, engineer features, scale, and save to parquet
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Batch Shapes
    inputs, targets, u_out = next(iter(train_loader))

    print(
        f"Batch shapes -> Inputs: {inputs.shape}, Targets: {targets.shape}, u_out: {u_out.shape}"
    )

    # Assertions
    # Shape: (Batch, Seq_Len, Features)
    expected_input_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    expected_target_shape = (Config.BATCH_SIZE, Config.SEQ_LEN)

    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"
    assert (
        u_out.shape == expected_target_shape
    ), f"u_out shape mismatch. Expected {expected_target_shape}, got {u_out.shape}"

    print("DataLoaders verified successfully.")

    # ==========================================
    # 4. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[4] Initializing Model and checking forward pass...")

    device = torch.device(Config.DEVICE)
    model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(device)

    # Move dummy batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(inputs)

    print(f"Prediction shape: {preds.shape}")
    assert (
        preds.shape == expected_target_shape
    ), f"Prediction shape mismatch. Expected {expected_target_shape}, got {preds.shape}"
    print("Model architecture verified successfully.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Initialize Trainer (it will pick up the cached debug data we just generated)
    trainer = Trainer()
    trainer.fit()

    # Verify model artifact
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Training failed to produce 'best_model.pth'")

    print("Training complete. Model saved.")

    # ==========================================
    # 6. Inference Preparation (Metadata Alignment)
    # ==========================================
    print("\n[6] Preparing Metadata for Debug Inference...")

    # In Debug mode, we only processed a subset of the test data.
    # The inference script expects metadata for the *entire* test set by default.
    # We must filter the metadata to match our debug subset to avoid length mismatches.

    # 1. Identify which breaths are in the processed test cache
    df_test_debug = pd.read_parquet(Config.CACHE_TEST_PATH)
    debug_breath_ids = df_test_debug["breath_id"].unique()

    # 2. Load full metadata
    full_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # 3. Filter metadata
    debug_test_meta = full_test_meta[
        full_test_meta["breath_id"].isin(debug_breath_ids)
    ].copy()

    # 4. Save to temporary location
    debug_meta_path = os.path.join(Config.WORKING_DIR, "test_debug.csv")
    debug_test_meta.to_csv(debug_meta_path, index=False)

    # 5. Point Config to this new metadata file
    Config.TEST_META = debug_meta_path
    print(f"Debug metadata created with {len(debug_test_meta)} rows.")

    # ==========================================
    # 7. Inference Execution
    # ==========================================
    print("\n[7] Running Inference...")

    # Run prediction using the trained model and the debug test set
    predict(load_cached_data=True)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Inference failed to produce 'submission.csv'")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check format
    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) == len(
        debug_test_meta
    ), f"Submission length {len(df_sub)} != Metadata length {len(debug_test_meta)}"

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
