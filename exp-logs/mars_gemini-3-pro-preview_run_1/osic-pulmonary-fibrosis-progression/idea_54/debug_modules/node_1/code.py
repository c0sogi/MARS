import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders, OSICDataset
from library.model import NSLHN
from library.train import Runner, LaplaceLogLikelihoodLoss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("OSIC Pulmonary Fibrosis Progression - Code Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Safety
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Modify Config to run a fast demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PATIENCE = 2

    # Ensure we use the working directory for outputs
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated for demo execution.")

    # ---------------------------------------------------------
    # 2. Metric Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Metric Logic...")

    # Test Case 1: Perfect Prediction
    # Delta = 0, Sigma = 100 (clipped to 100)
    # Metric = - (sqrt(2)*0/100) - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    print(f"Test Case: True={y_true}, Pred={y_pred}, Sigma={sigma}")
    print(f"Calculated Score: {score:.4f}, Expected: {expected_score:.4f}")

    assert np.isclose(score, expected_score, atol=1e-4), "Metric calculation mismatch!"
    print("Metric verification passed.")

    # ---------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing Data Pipeline (Debug Mode)...")

    # Use debug=True to load only 32 samples
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Fetch one batch to inspect
    batch = next(iter(train_loader))

    # Verify keys
    required_keys = [
        "axial",
        "coronal",
        "tabular",
        "fvc",
        "base_fvc",
        "week",
        "base_week",
    ]
    for k in required_keys:
        assert k in batch, f"Missing key in batch: {k}"

    # Verify Shapes
    # Images: (B, 3, 224, 224)
    B = batch["axial"].shape[0]
    assert batch["axial"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect Axial shape: {batch['axial'].shape}"
    assert batch["coronal"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect Coronal shape: {batch['coronal'].shape}"

    # Tabular: (B, 8) -> Weeks, Percent, Age, Sex(2), Smoke(3)
    assert batch["tabular"].shape == (
        B,
        8,
    ), f"Incorrect Tabular shape: {batch['tabular'].shape}"

    print(f"Batch verification passed. Batch size: {B}")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = NSLHN().to(device)

    # Move batch to device
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    base_fvc = batch["base_fvc"].to(device)
    week = batch["week"].to(device)
    base_week = batch["base_week"].to(device)

    # Forward Pass
    pred_fvc, pred_sigma = model(axial, coronal, tabular, base_fvc, week, base_week)

    print(f"Output FVC Shape: {pred_fvc.shape}")
    print(f"Output Sigma Shape: {pred_sigma.shape}")

    assert pred_fvc.shape == (B,), "Prediction shape mismatch"
    assert pred_sigma.shape == (B,), "Sigma shape mismatch"

    # Check constraints
    # Sigma should be positive (Softplus used in model)
    assert (pred_sigma > 0).all(), "Sigma must be positive"

    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Loop Integration
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Loop (Runner)...")

    # Initialize Runner with debug=True
    runner = Runner(debug=True)

    # Run training
    # This will run for Config.EPOCHS (set to 2) on the small subset
    runner.train()

    # Verify Checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint successfully created at: {best_model_path}")
    else:
        # If validation score never improved (unlikely with random init), it might not save.
        # But usually it saves at least once.
        print(
            "Warning: No best model checkpoint found (might be due to extremely short run)."
        )

    # ---------------------------------------------------------
    # 6. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[Step 6] Inference on Validation Batch...")

    model.eval()
    with torch.no_grad():
        # Re-use the batch from earlier (or fetch new one)
        # Note: In real inference, we use test_loader
        val_batch = next(iter(val_loader))

        v_axial = val_batch["axial"].to(device)
        v_coronal = val_batch["coronal"].to(device)
        v_tabular = val_batch["tabular"].to(device)
        v_base_fvc = val_batch["base_fvc"].to(device)
        v_week = val_batch["week"].to(device)
        v_base_week = val_batch["base_week"].to(device)
        v_true = val_batch["fvc"].to(device)

        p_fvc, p_sigma = model(
            v_axial, v_coronal, v_tabular, v_base_fvc, v_week, v_base_week
        )

        # Calculate metric
        final_metric = laplace_log_likelihood_metric(v_true, p_fvc, p_sigma)

        print("Sample Predictions:")
        for i in range(min(3, len(p_fvc))):
            print(
                f"  True: {v_true[i].item():.1f} | Pred: {p_fvc[i].item():.1f} | Conf: {p_sigma[i].item():.1f}"
            )

        print(f"Batch Metric Score: {final_metric:.4f}")

    print("\n" + "=" * 50)
    print("Demo execution completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
