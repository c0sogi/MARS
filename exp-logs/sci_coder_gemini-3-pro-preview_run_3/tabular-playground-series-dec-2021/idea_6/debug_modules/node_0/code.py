import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.model import DCNResNet
from library.train_eval import run_training


def main():
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Initializing Demo Task...")

    # Override Config to use a separate demo directory and update dependent paths
    # Note: We must update these manually because they are class attributes
    # evaluated at import time in the library.
    Config.WORKING_DIR = "./working/demo_task"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    Config.CACHE_TRAIN_X = os.path.join(Config.WORKING_DIR, "train_X.npy")
    Config.CACHE_TRAIN_Y = os.path.join(Config.WORKING_DIR, "train_y.npy")
    Config.CACHE_VAL_X = os.path.join(Config.WORKING_DIR, "val_X.npy")
    Config.CACHE_VAL_Y = os.path.join(Config.WORKING_DIR, "val_y.npy")
    Config.CACHE_TEST_X = os.path.join(Config.WORKING_DIR, "test_X.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    # Ensure directories exist
    Config.setup()

    # 2. Unit Test: Model Architecture
    # ---------------------------------------------------------
    print("\n[Test 1] Verifying Model Architecture...")

    # Define dummy parameters
    batch_size = 16
    input_dim = 64
    num_classes = 7

    # Instantiate model
    model = DCNResNet(
        input_dim=input_dim,
        hidden_dim=32,
        num_res_blocks=2,
        num_cross_layers=2,
        num_classes=num_classes,
        dropout_rate=0.1,
    )

    # Create dummy input
    dummy_input = torch.randn(batch_size, input_dim)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Model output shape mismatch. Expected {(batch_size, num_classes)}, got {output.shape}"

    print("Model forward pass successful.")

    # 3. Integration Test: Training Pipeline
    # ---------------------------------------------------------
    print("\n[Test 2] Running Training Pipeline (Fast Mode)...")

    # We run the pipeline with:
    # - 1 Epoch
    # - Large batch size (for speed)
    # - Max 5000 training samples (to verify training loop works without waiting for full epoch)
    # - load_cached_data=False to demonstrate raw data processing logic

    try:
        run_training(
            epochs=1, batch_size=2048, load_cached_data=False, max_train_samples=5000
        )
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise e

    # 4. Verification: Output Files
    # ---------------------------------------------------------
    print("\n[Test 3] Verifying Submission Output...")

    submission_path = Config.SUBMISSION_FILE

    # Check file existence
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    # Load submission
    df_sub = pd.read_csv(submission_path)

    # Check shape (Test set is 400,000 rows)
    expected_rows = 400000
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    # Check columns
    required_cols = ["Id", "Cover_Type"]
    if not all(col in df_sub.columns for col in required_cols):
        raise ValueError(f"Submission missing required columns: {required_cols}")

    # Check ID type and Target range
    if not pd.api.types.is_integer_dtype(df_sub["Id"]):
        raise TypeError("Id column must be integer.")

    # Cover_Type should be between 1 and 7
    min_class = df_sub["Cover_Type"].min()
    max_class = df_sub["Cover_Type"].max()
    if min_class < 1 or max_class > 7:
        raise ValueError(
            f"Predictions out of range [1, 7]. Found min={min_class}, max={max_class}"
        )

    print("Submission file verified successfully.")
    print(f"Sample predictions:\n{df_sub.head()}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
