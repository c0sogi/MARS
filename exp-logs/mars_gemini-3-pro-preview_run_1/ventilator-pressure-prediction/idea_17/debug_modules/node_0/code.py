import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data
from library.model import WideStateNet
from library.loss import CompositeMaskedL1Loss
from library.train import train_model
from library.inference import predict


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # --------------------------------------------------------------------------
    print("Step 1: Configuring environment for fast demonstration...")

    # Override Config defaults to ensure speed
    Config.EXPERIMENT_NAME = "demo_verification_run"
    Config.DEBUG = True  # Triggers data subsampling (100 breaths train, 50 val)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.USE_CACHE = (
        False  # Force fresh processing to verify feature engineering logic
    )
    Config.BATCH_SIZE = 16  # Small batch size for the small debug dataset

    # Re-setup directories based on new experiment name
    Config.setup()

    # Clean up previous run if exists to ensure fresh start
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\nStep 2: Verifying Data Loading and Processing...")

    # Load data (this will trigger feature engineering and scaling)
    train_loader, val_loader, test_loader, feature_names = prepare_data(
        load_cached_data=False
    )

    # Verify Feature Names
    print(f"Generated {len(feature_names)} features.")
    assert len(feature_names) > 0, "Feature list is empty."
    assert "u_in_lag1" in feature_names, "Lag features missing."
    assert "u_in_diff1" in feature_names, "Difference features missing."

    # Verify DataLoader Batch Structure
    batch = next(iter(train_loader))
    x, u_out, y = batch["x"], batch["u_out"], batch["y"]

    print(f"Batch shapes - X: {x.shape}, u_out: {u_out.shape}, y: {y.shape}")

    # Assertions for shapes
    # Expected: (Batch_Size, 80, Num_Features)
    assert x.shape == (
        Config.BATCH_SIZE,
        80,
        len(feature_names),
    ), "Incorrect input shape."
    assert y.shape == (Config.BATCH_SIZE, 80), "Incorrect target shape."
    assert u_out.shape == (Config.BATCH_SIZE, 80), "Incorrect u_out shape."

    print("Data pipeline verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\nStep 3: Verifying Model Architecture...")

    model = WideStateNet(input_dim=len(feature_names), feature_names=feature_names)
    model.to(device)

    # Move batch to device
    x = x.to(device)
    u_out = u_out.to(device)
    y = y.to(device)

    # Forward Pass
    final_pred, aux_pred = model(x, u_out)

    print(f"Model Output Shape: {final_pred.shape}")

    # Assertions
    assert final_pred.shape == (Config.BATCH_SIZE, 80), "Model output shape mismatch."
    if Config.USE_AUXILIARY_HEAD:
        assert aux_pred.shape == (
            Config.BATCH_SIZE,
            80,
        ), "Auxiliary output shape mismatch."

    print("Model architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\nStep 4: Verifying Loss Function...")

    criterion = CompositeMaskedL1Loss()
    loss = criterion((final_pred, aux_pred), y, u_out)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss is negative."

    print("Loss function verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\nStep 5: Executing Training Loop (1 Epoch)...")

    # train_model() handles the loop, saving, and logging internally.
    # It uses the Config we set up earlier.
    train_model()

    # Verify artifacts
    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved."

    print("Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Inference Execution
    # --------------------------------------------------------------------------
    print("\nStep 6: Executing Inference...")

    # predict() loads the saved model and generates submission.csv
    predict(load_cached_data=True)  # Use cache generated during training step

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found."

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    assert list(sub_df.columns) == ["id", "pressure"], "Incorrect submission columns."
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    # In Debug mode, we processed 50 breaths * 80 steps = 4000 rows
    expected_rows = 50 * 80
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}."

    print("Inference verification passed.")
    print("\n=== All Demonstration Steps Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
