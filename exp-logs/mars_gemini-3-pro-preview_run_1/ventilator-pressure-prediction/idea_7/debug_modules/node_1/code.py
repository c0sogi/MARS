import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import get_dataloaders
from library.model import VentilatorModel
from library.engine import Engine


def main():
    # 1. Setup and Configuration Override for Demo Speed
    print("=== Setting up Demo Configuration ===")
    seed_everything(42)

    # Modify Config for a fast, debug run
    Config.EXP_ID = "demo_execution"
    Config.DEBUG = True  # Use small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 64  # Smaller batch size

    # Re-run setup to create the new experiment directories
    # We need to update paths that rely on EXP_ID
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_ID)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading and Verification
    print("\n=== Loading Data ===")
    # load_cached_data=False forces reprocessing since we changed EXP_ID/DEBUG settings
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print("Verifying Data Loaders...")
    # Fetch one batch to verify shapes
    x, y, u_out = next(iter(train_loader))

    # Check shapes
    # x: (Batch, Seq_Len, Features)
    # y: (Batch, Seq_Len)
    # u_out: (Batch, Seq_Len)
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Mask shape: {u_out.shape}")

    assert x.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert x.shape[1] == Config.SEQ_LEN, "Sequence length mismatch"
    assert y.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Target shape mismatch"
    assert u_out.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "u_out shape mismatch"
    print("Data Loader verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n=== Initializing Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VentilatorModel().to(device)

    print("Verifying Forward Pass...")
    x = x.to(device)
    with torch.no_grad():
        preds = model(x)

    # Model output should be (Batch, Seq_Len, 1)
    print(f"Prediction shape: {preds.shape}")
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), "Model output shape mismatch"
    print("Model forward pass verification passed.")

    # 4. Metric Logic Verification
    print("\n=== Verifying Metric Logic ===")
    # Test case:
    # u_out = 0 (Inspiratory) -> Error should count
    # u_out = 1 (Expiratory) -> Error should be ignored

    y_true_test = torch.tensor([10.0, 100.0])
    y_pred_test = torch.tensor([12.0, 200.0])  # Error: 2.0 and 100.0
    u_out_test = torch.tensor([0.0, 1.0])  # Only first element matters

    # Expected MAE: |12 - 10| = 2.0. The second error (100.0) is masked out.
    mae = compute_metric(y_pred_test, y_true_test, u_out_test)
    print(f"Calculated MAE: {mae}")

    assert abs(mae - 2.0) < 1e-6, f"Metric calculation failed. Expected 2.0, got {mae}"
    print("Metric logic verification passed.")

    # 5. Training Loop Execution
    print("\n=== Starting Training Loop (Demo) ===")
    engine = Engine(model, device)

    # Run fit (1 epoch as per config override)
    engine.fit(train_loader, val_loader)

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint not found after training"
    print(f"Training complete. Model saved to {Config.MODEL_PATH}")

    # 6. Inference and Submission
    print("\n=== Generating Submission ===")
    engine.generate_submission(test_loader)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Check columns
    assert list(df_sub.columns) == ["id", "pressure"], "Submission columns mismatch"

    # In DEBUG mode, we only use a subset of the test data.
    # We verify that the number of rows matches the number of IDs in the test loader.
    # Note: test_loader drops nothing, but batching might leave a remainder.
    # However, get_data processes specific amounts in DEBUG mode.

    # Calculate expected length based on the test dataset size
    expected_len = len(test_loader.dataset) * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print("Submission verification passed.")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
