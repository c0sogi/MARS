import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.data import DataProcessor, get_dataloaders, VentilatorDataset
from library.model import MultiScaleSE_LSTM
from library.trainer import Trainer
from library.utils import set_seed, compute_metric


# ------------------------------------------------------------------------------
# 1. Configuration for Demo
# ------------------------------------------------------------------------------
class DemoConfig(Config):
    """
    Modified configuration for a fast demonstration run.
    """

    # Enable Debug mode to use a tiny subset of data (200 breaths)
    DEBUG = True

    # Reduce training duration
    EPOCHS = 1
    BATCH_SIZE = 16

    # Reduce model complexity for speed
    LSTM_HIDDEN = 64
    LSTM_LAYERS = 1
    CNN_FILTERS = 16
    SE_RATIO = 4

    # Use a separate directory for demo artifacts to avoid overwriting real work
    WORKING_DIR = "./working/demo_execution"

    # Update paths based on new working dir
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    SCALER_CENTER_PATH = os.path.join(WORKING_DIR, "scaler_center.npy")
    SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "scaler_scale.npy")

    # Ensure the demo directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)


def run_demo():
    print("=== Starting Ventilator Prediction Library Demo ===")

    # Set seed for reproducibility
    set_seed(DemoConfig.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # --------------------------------------------------------------------------
    print("\n[1/5] Testing DataProcessor...")
    processor = DataProcessor(DemoConfig)

    # Load and process training data (DEBUG mode limits this to 200 breaths)
    # We force load_cached_data=False to demonstrate the engineering pipeline
    print("Processing training data...")
    X_train, y_train, u_out_train, ids_train = processor.load_data(
        "train", load_cached_data=False
    )

    # Validation assertions
    print("Validating data shapes...")
    # Shape should be (N_breaths, Seq_Len, N_Features)
    assert X_train.ndim == 3, f"Expected 3D input, got {X_train.ndim}"
    assert (
        X_train.shape[1] == DemoConfig.SEQ_LEN
    ), f"Expected sequence length {DemoConfig.SEQ_LEN}, got {X_train.shape[1]}"
    assert (
        X_train.shape[2] == DemoConfig.INPUT_DIM
    ), f"Expected {DemoConfig.INPUT_DIM} features, got {X_train.shape[2]}"

    # Check targets
    assert y_train.shape == (
        X_train.shape[0],
        DemoConfig.SEQ_LEN,
    ), "Target shape mismatch"
    assert u_out_train.shape == y_train.shape, "Control input shape mismatch"

    # Check if scaler artifacts were created
    assert os.path.exists(DemoConfig.SCALER_CENTER_PATH), "Scaler center file not saved"
    assert os.path.exists(DemoConfig.SCALER_SCALE_PATH), "Scaler scale file not saved"

    print(f"Data processed successfully. Shape: {X_train.shape}")

    # --------------------------------------------------------------------------
    # 3. Metric Calculation Demonstration
    # --------------------------------------------------------------------------
    print("\n[2/5] Testing Metric Calculation...")
    # Create dummy data
    # Case: 2 time steps.
    # Step 0: Inspiratory (u_out=0), Pred=10, Target=12 -> Error=2
    # Step 1: Expiratory (u_out=1), Pred=100, Target=200 -> Error=Ignored
    dummy_preds = np.array([10.0, 100.0])
    dummy_targets = np.array([12.0, 200.0])
    dummy_u_out = np.array([0, 1])

    mae = compute_metric(dummy_preds, dummy_targets, dummy_u_out)
    print(f"Computed MAE: {mae}")

    assert np.isclose(mae, 2.0), f"Metric calculation failed. Expected 2.0, got {mae}"
    print("Metric logic verified.")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Demonstration
    # --------------------------------------------------------------------------
    print("\n[3/5] Testing Model Architecture...")
    model = MultiScaleSE_LSTM(DemoConfig)

    # Move to CPU for shape check
    model.to("cpu")
    model.eval()

    # Create a dummy batch: (Batch=2, Seq=80, Feat=Input_Dim)
    dummy_input = torch.randn(2, DemoConfig.SEQ_LEN, DemoConfig.INPUT_DIM)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Output should be (Batch, Seq, 1)
    assert output.shape == (2, DemoConfig.SEQ_LEN, 1), "Model output shape mismatch"
    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[4/5] Testing Trainer (Fit Loop)...")
    trainer = Trainer(DemoConfig)

    # Run training
    # We use load_cached_data=True because we generated the cache in step [1/5]
    trainer.fit(load_cached_data=True)

    # Verify model artifact
    assert os.path.exists(
        DemoConfig.MODEL_PATH
    ), "Model file was not saved after training"
    print("Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Inference Demonstration
    # --------------------------------------------------------------------------
    print("\n[5/5] Testing Inference (Predict Loop)...")

    # Run prediction on test set
    # Note: load_cached_data=False ensures we process test data from scratch for this demo
    trainer.predict(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(DemoConfig.SUBMISSION_PATH), "Submission file not found"

    # Check submission format
    sub_df = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission columns incorrect"
    assert len(sub_df) > 0, "Submission file is empty"

    # In DEBUG mode, we only process a subset of breaths.
    # The test set in DEBUG mode will be small (200 breaths * 80 steps = 16000 rows max)
    # Just verifying it ran is sufficient.

    print("Inference completed successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
