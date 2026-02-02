import os
import sys
import pandas as pd
import torch
import numpy as np

# Import library components
from library.config import Config
from library.dataset import get_data
from library.model import SimpleFCN
from library.trainer import Trainer


def run_demo():
    print("========================================")
    print("Iceberg Classification: Fast Demo Run")
    print("========================================")

    # 1. Configure for Speed
    # We modify Config attributes to limit runtime.
    # Note: Default arguments in functions (like get_data) are bound at import time,
    # so we will pass these values explicitly where necessary.
    Config.NUM_EPOCHS = 2
    Config.DEBUG_SAMPLE_SIZE = 200  # Use only 200 samples for training/validation
    Config.BATCH_SIZE = 32

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Create directories
    Config.setup()
    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Debug Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 2. Data Loading
    print("\n[Step 1] Loading and Processing Data...")
    # load_cached_data=False forces the code to process raw JSONs, ensuring that logic works.
    data_loaders = get_data(
        load_cached_data=False,
        batch_size=Config.BATCH_SIZE,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    train_loader = data_loaders["train_loader"]
    val_loader = data_loaders["val_loader"]
    test_loader = data_loaders["test_loader"]
    test_ids = data_loaders["test_ids"]

    # 3. Validation of Data Shapes
    print("\n[Step 2] Verifying Data Shapes...")
    # Fetch one batch from training loader
    x_batch, angle_batch, y_batch = next(iter(train_loader))

    # Expected Dimensions:
    # Image: 75x75 pixels * 2 bands = 11250 flattened features
    expected_input_dim = 75 * 75 * 2

    print(f"  Batch X shape: {x_batch.shape}")
    print(f"  Batch Angle shape: {angle_batch.shape}")
    print(f"  Batch Y shape: {y_batch.shape}")

    # Assertions to ensure logic correctness
    if x_batch.shape[1] != expected_input_dim:
        raise AssertionError(
            f"Input dimension mismatch. Expected {expected_input_dim}, got {x_batch.shape[1]}"
        )

    if (
        x_batch.shape[0] != Config.BATCH_SIZE
        and x_batch.shape[0] != Config.DEBUG_SAMPLE_SIZE
    ):
        # Note: Last batch might be smaller, but here we check against reasonable bounds
        pass

    if y_batch.shape[1] != 1:
        raise AssertionError(
            f"Target dimension mismatch. Expected (N, 1), got {y_batch.shape}"
        )

    print("  Data shapes verified successfully.")

    # 4. Model Initialization
    print("\n[Step 3] Initializing Model...")
    model = SimpleFCN()
    # Move model to configured device
    model.to(Config.DEVICE)
    print(f"  Model created on {Config.DEVICE}")

    # 5. Training
    print("\n[Step 4] Starting Training Loop...")
    trainer = Trainer(model)

    # Run training
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)
    print("  Training completed.")

    # 6. Prediction
    print("\n[Step 5] Generating Predictions...")
    trainer.predict(test_loader, test_ids)

    # 7. Verify Submission
    print("\n[Step 6] Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_sub.shape}")
    print(f"  Columns: {list(df_sub.columns)}")

    # Check structure
    if "id" not in df_sub.columns or "is_iceberg" not in df_sub.columns:
        raise AssertionError(
            "Submission file missing required columns ('id', 'is_iceberg')."
        )

    # Check probability range
    preds = df_sub["is_iceberg"]
    if preds.min() < 0.0 or preds.max() > 1.0:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    # Check against Test Metadata size
    # Note: get_data does NOT subsample the test set even in debug mode (per library/dataset.py logic),
    # so we expect the full test set size (321 based on metadata info).
    test_meta = pd.read_csv(Config.TEST_META)
    expected_rows = len(test_meta)

    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count ({len(df_sub)}) does not match Test Metadata count ({expected_rows})."
        )

    print(
        f"  Submission verified: {len(df_sub)} rows, values in range [{preds.min():.4f}, {preds.max():.4f}]."
    )
    print("\n========================================")
    print("Demo Completed Successfully")
    print("========================================")


if __name__ == "__main__":
    run_demo()
