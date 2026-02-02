import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import VentilatorModel, loss_fn
from library.train import train


def main():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("1. Setting up configuration for fast execution...")

    # Define a specific experiment ID for this demo to isolate outputs
    Config.EXPERIMENT_ID = "demo_execution"

    # Setup directories
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_ID)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = Config.WORKING_DIR  # Save submission in working dir
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Clean up previous runs
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Override Hyperparameters for Speed
    Config.DEBUG = True  # Use small subset (100 breaths)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Handle Sample Submission Mismatch for Debug Mode
    # The model generates predictions for the debug subset (8000 rows),
    # but the original sample_submission is full size. We must truncate it.
    print("   Creating truncated sample_submission for debug mode...")
    full_sample_sub = pd.read_csv("./input/sample_submission.csv")
    # Debug mode in data.py takes first 100 breaths * 80 steps = 8000 rows
    debug_sample_sub = full_sample_sub.iloc[:8000].copy()
    truncated_sub_path = os.path.join(Config.WORKING_DIR, "sample_submission.csv")
    debug_sample_sub.to_csv(truncated_sub_path, index=False)
    Config.SAMPLE_SUBMISSION = truncated_sub_path

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    print("   Configuration complete.\n")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("2. Verifying Data Pipeline...")

    # Force load_cached_data=False to trigger the feature engineering and preprocessing logic
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch a single batch
    X_batch, y_batch, u_out_batch = next(iter(train_loader))

    print(
        f"   Batch Shapes -> X: {X_batch.shape}, y: {y_batch.shape}, u_out: {u_out_batch.shape}"
    )

    # Assertions
    assert X_batch.shape[0] == Config.BATCH_SIZE
    assert X_batch.shape[1] == 80  # Sequence length
    assert y_batch.shape == (Config.BATCH_SIZE, 80)
    assert u_out_batch.shape == (Config.BATCH_SIZE, 80)

    input_dim = X_batch.shape[-1]
    print(f"   Input Features: {input_dim}")
    print("   Data Pipeline verified.\n")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("3. Verifying Model Architecture...")

    model = VentilatorModel(input_dim=input_dim).to(device)

    # Create dummy input on device
    dummy_input = torch.randn(2, 80, input_dim).to(device)

    # Forward pass
    pred, aux_pred = model(dummy_input)

    print(f"   Output Shapes -> Pred: {pred.shape}, Aux: {aux_pred.shape}")

    # Assertions
    # Output should be (Batch, Seq, 1)
    assert pred.shape == (2, 80, 1), f"Expected (2, 80, 1), got {pred.shape}"
    assert aux_pred.shape == (2, 80, 1), f"Expected (2, 80, 1), got {aux_pred.shape}"

    print("   Model Architecture verified.\n")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("4. Verifying Loss Function Logic...")

    # Case A: Perfect Prediction (Loss should be 0)
    t_pred = torch.ones(2, 80, 1).to(device)
    t_target = torch.ones(2, 80).to(device)
    t_uout = torch.zeros(2, 80).to(device)  # All inspiratory (mask=1)

    loss_val = loss_fn(t_pred, t_target, t_uout)
    assert loss_val.item() == 0.0, f"Loss should be 0.0, got {loss_val.item()}"

    # Case B: Error in Expiratory Phase (Should be masked out)
    # Prediction 0, Target 1, but u_out 1 (expiratory -> mask=0)
    t_pred_bad = torch.zeros(2, 80, 1).to(device)
    t_uout_exp = torch.ones(2, 80).to(device)

    loss_masked = loss_fn(t_pred_bad, t_target, t_uout_exp)
    assert (
        loss_masked.item() == 0.0
    ), f"Loss should be masked to 0.0, got {loss_masked.item()}"

    print("   Loss Function verified.\n")

    # -------------------------------------------------------------------------
    # 5. Integration Test: Full Training Loop
    # -------------------------------------------------------------------------
    print("5. Running Full Training Cycle (Integration Test)...")

    # We use the high-level train function from library.train
    # This tests the optimizer, scheduler, loop, validation, and checkpointing
    try:
        train(
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            learning_rate=1e-3,
            debug=Config.DEBUG,
            load_cached_data=True,  # Use the cache we generated in step 2
        )
        print("   Training cycle completed successfully.\n")
    except Exception as e:
        print(f"   Training cycle failed with error: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 6. Artifact Verification
    # -------------------------------------------------------------------------
    print("6. Verifying Output Artifacts...")

    # Check Model Checkpoint
    if os.path.exists(Config.MODEL_PATH):
        print(f"   [OK] Model checkpoint found at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model checkpoint missing at {Config.MODEL_PATH}")

    # Check Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   [OK] Submission file found at {Config.SUBMISSION_PATH}")
        print(f"   Submission Shape: {sub_df.shape}")

        # Validate submission format
        assert list(sub_df.columns) == ["id", "pressure"], "Invalid submission columns"
        assert (
            sub_df.shape[0] == 8000
        ), f"Expected 8000 rows (Debug Mode), got {sub_df.shape[0]}"
    else:
        raise FileNotFoundError(f"Submission file missing at {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
