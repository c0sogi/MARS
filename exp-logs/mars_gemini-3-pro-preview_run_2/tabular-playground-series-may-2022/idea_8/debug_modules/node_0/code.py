import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders, decompose_f27
from library.model import WideDeepResFunnel
from library.train_eval import train_model, set_seed


def run_demo():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # --------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a small subset of data (5000 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small enough for quick CPU/GPU execution

    # Reduce training duration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128  # Smaller batch size for the small subset

    # Use a separate cache file for this demo to avoid overwriting/using production cache
    Config.PROCESSED_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "processed_data_demo.npz"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure clean slate for demo artifacts
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        os.remove(Config.PROCESSED_DATA_PATH)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Samples: {Config.DEBUG_SAMPLES}")

    # --------------------------------------------------------------------------
    # 2. Verify Feature Engineering Logic
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying feature engineering (decompose_f27)...")

    # Test case: 'ABCDEFGHIJ' -> Should map to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_series = pd.Series(["ABCDEFGHIJ", "ZZZZZZZZZZ"])
    decomposed = decompose_f27(test_series)

    # Assertions
    assert isinstance(decomposed, np.ndarray), "Output should be a numpy array"
    assert decomposed.shape == (
        2,
        10,
    ), f"Expected shape (2, 10), got {decomposed.shape}"
    assert decomposed.dtype == np.int32, "Expected int32 dtype"

    # Check values for 'ABCDEFGHIJ'
    expected_abc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert np.array_equal(
        decomposed[0], expected_abc
    ), f"Mapping incorrect. Got {decomposed[0]}"

    # Check values for 'ZZZZZZZZZZ' (Z=26)
    expected_z = np.full(10, 26)
    assert np.array_equal(
        decomposed[1], expected_z
    ), f"Mapping incorrect. Got {decomposed[1]}"

    print("Feature engineering logic verified.")

    # --------------------------------------------------------------------------
    # 3. Verify Data Loading and Shapes
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying DataLoaders and Batch Shapes...")

    # Load data (this will trigger processing since we deleted the demo cache)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    numeric = batch["numeric"]
    categorical = batch["categorical"]
    targets = batch["target"]

    # Assertions
    print(f"Batch keys: {batch.keys()}")

    # Check Numeric Features: (Batch, 30)
    assert numeric.dim() == 2, "Numeric data should be 2D"
    assert (
        numeric.shape[1] == Config.NUM_CONTINUOUS_FEATURES
    ), f"Expected {Config.NUM_CONTINUOUS_FEATURES} numeric features, got {numeric.shape[1]}"

    # Check Categorical Features: (Batch, 10)
    assert categorical.dim() == 2, "Categorical data should be 2D"
    assert (
        categorical.shape[1] == Config.F_27_SEQ_LEN
    ), f"Expected seq len {Config.F_27_SEQ_LEN}, got {categorical.shape[1]}"

    # Check Targets: (Batch,)
    assert targets.dim() == 1, "Targets should be 1D"
    assert (
        targets.shape[0] == numeric.shape[0]
    ), "Batch size mismatch between inputs and targets"

    print("Data loading verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture and Forward Pass
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = WideDeepResFunnel().to(device)

    # Move batch to device
    num_gpu = numeric.to(device)
    cat_gpu = categorical.to(device)

    # Forward pass
    final_logits, aux1, aux2 = model(num_gpu, cat_gpu)

    # Assertions on outputs
    batch_size = num_gpu.size(0)

    # Check Final Logits
    assert final_logits.shape == (
        batch_size,
        1,
    ), f"Expected final_logits shape {(batch_size, 1)}, got {final_logits.shape}"

    # Check Aux Logits (Deep Supervision)
    assert aux1.shape == (batch_size, 1), "Aux1 shape mismatch"
    assert aux2.shape == (batch_size, 1), "Aux2 shape mismatch"

    print("Model forward pass verified.")

    # --------------------------------------------------------------------------
    # 5. Execute Full Training Pipeline
    # --------------------------------------------------------------------------
    print("\n[Step 5] Executing Training Pipeline (train_model)...")

    # This function encapsulates the loop, validation, and prediction
    # It uses the Config settings we modified in Step 1
    train_model()

    print("Training pipeline execution complete.")

    # --------------------------------------------------------------------------
    # 6. Verify Submission Artifacts
    # --------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns 'id' or 'target'"

    # Check length
    # In DEBUG mode, test set is also sliced to DEBUG_SAMPLES
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} rows in submission, got {len(df_sub)}"

    # Check value range (probabilities)
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print(f"Submission verified. Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    # Set seed for reproducibility of the demo script itself
    set_seed(42)
    run_demo()
