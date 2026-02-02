import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.features import prepare_dataset
from library.dataset import VentilatorDataset
from library.model import GraduatedCapacityNetwork
from library.loss import MaskedAuxiliaryLoss
from library.train import Trainer


def run_demo():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Library ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config for speed and debugging
    Config.DEBUG = True  # Use a small subset of data (200 breaths)
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Re-run setup to ensure the new experiment directory exists
    Config.setup()

    # Clean up any previous demo run in this directory to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        # We only clean cache files to force re-processing for the demo
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".npy"):
                os.remove(os.path.join(Config.WORKING_DIR, f))

    seed_everything(Config.SEED)
    print(f"    Experiment Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification (prepare_dataset)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline (prepare_dataset)...")

    # We force load_cached_data=False to test the feature engineering logic
    data = prepare_dataset(split="train", load_cached_data=False)

    X = data["X"]
    y = data["y"]
    u_out = data["u_out"]
    ids = data["ids"]

    print(f"    Processed Data Shapes:")
    print(f"      X: {X.shape} (Breaths, TimeSteps, Features)")
    print(f"      y: {y.shape} (Breaths, TimeSteps)")
    print(f"      u_out: {u_out.shape}")

    # Assertions
    assert X.ndim == 3, "X should be 3-dimensional"
    assert X.shape[1] == 80, "Time dimension should be 80 steps per breath"
    assert y is not None, "Training data should have targets"
    assert X.shape[0] == y.shape[0] == u_out.shape[0], "Sample counts must match"

    # Check for NaNs
    assert not np.isnan(X).any(), "Feature matrix X contains NaNs"
    assert not np.isnan(y).any(), "Target vector y contains NaNs"

    print("    Data Pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset Class Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying PyTorch Dataset (VentilatorDataset)...")

    # Initialize dataset (will use the cached data generated above)
    dataset = VentilatorDataset(split="train", load_cached_data=True)

    # Fetch one sample
    sample_X, sample_u_out, sample_y, sample_id = dataset[0]

    # Check types and shapes
    assert isinstance(sample_X, torch.Tensor), "Output should be a Tensor"
    assert sample_X.dtype == torch.float32, "Features should be float32"
    assert sample_u_out.shape == (80,), "u_out shape mismatch"
    assert sample_y.shape == (80,), "Target shape mismatch"

    print(f"    Dataset sample retrieved successfully.")
    print(f"    Input Feature Dimension: {sample_X.shape[-1]}")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (GraduatedCapacityNetwork)...")

    input_dim = sample_X.shape[-1]
    model = GraduatedCapacityNetwork(input_dim=input_dim)
    model.eval()

    # Create dummy batch: (Batch=4, Length=80, Features=input_dim)
    dummy_input = torch.randn(4, 80, input_dim)

    # Forward pass
    with torch.no_grad():
        final_pred, aux_pred = model(dummy_input)

    print(f"    Model Output Shapes:")
    print(f"      Final Pred: {final_pred.shape}")
    if aux_pred is not None:
        print(f"      Aux Pred:   {aux_pred.shape}")

    # Assertions
    assert final_pred.shape == (4, 80), "Final prediction shape incorrect"
    if Config.USE_AUX_HEAD:
        assert aux_pred.shape == (4, 80), "Aux prediction shape incorrect"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Loss Function (MaskedAuxiliaryLoss)...")

    criterion = MaskedAuxiliaryLoss(aux_weight=0.5)

    # Create synthetic data
    # Case: Perfect prediction where u_out=0, Error where u_out=1
    # Since u_out=1 is masked out, loss should be 0.

    batch_size = 2
    length = 80

    # Targets: All 10s
    targets = torch.full((batch_size, length), 10.0)

    # Predictions: All 10s (Perfect)
    preds_perfect = torch.full((batch_size, length), 10.0)

    # Predictions: Error only in expiratory phase
    # u_out: First 30 steps 0 (Inspiratory), Last 50 steps 1 (Expiratory)
    u_out = torch.zeros((batch_size, length))
    u_out[:, 30:] = 1.0

    preds_error_expiratory = preds_perfect.clone()
    preds_error_expiratory[:, 40:] = 100.0  # Huge error in masked region

    # Calculate Loss
    loss = criterion((preds_error_expiratory, preds_error_expiratory), targets, u_out)

    print(f"    Calculated Loss (Expected ~0.0): {loss.item()}")

    # Assert loss is effectively zero (floating point tolerance)
    assert loss.item() < 1e-6, "Loss should be zero when error is only in masked region"

    # Case: Error in inspiratory phase
    preds_error_inspiratory = preds_perfect.clone()
    # Add error of 1.0 to first 10 steps (where u_out=0)
    preds_error_inspiratory[:, :10] = 11.0

    # Manual Calc:
    # Total valid steps (u_out=0) = 30 * 2 (batch) = 60 steps
    # Error sum = 1.0 * 10 * 2 (batch) = 20.0
    # MAE = 20 / 60 = 0.3333
    # Total Loss = MAE_final + 0.5 * MAE_aux = 0.3333 + 0.5 * 0.3333 = 0.5

    loss_insp = criterion(
        (preds_error_inspiratory, preds_error_inspiratory), targets, u_out
    )
    print(f"    Calculated Loss with Error (Expected ~0.5): {loss_insp.item():.4f}")

    assert (
        abs(loss_insp.item() - 0.5) < 1e-4
    ), f"Loss mismatch. Got {loss_insp.item()}, expected 0.5"

    print("    Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (Trainer.fit)...")

    # Initialize Trainer
    # This will reload datasets (using cache) and initialize model/optimizer
    trainer = Trainer()

    # Run training
    # We expect this to run for 2 epochs on the small subset
    trainer.fit()

    # Verify model was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"    Model successfully saved to: {Config.MODEL_PATH}")
    else:
        raise AssertionError("Model file was not created after training.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
