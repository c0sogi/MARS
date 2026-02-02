import pandas as pd
import numpy as np
import torch
import os
import shutil
import sys

# Import library components
from library.config import Config
from library.data_utils import get_transformed_data
from library.model import DKRHNet
from library.train_utils import MaskedL1Loss, train_model


def setup_demo_environment():
    """
    Creates a temporary working environment with subset data for speed.
    """
    print("--- Setting up Demo Environment ---")

    # Define paths
    demo_input_dir = "./working/demo_input"
    demo_working_dir = "./working/demo_working"
    demo_submission_dir = "./working/demo_submission"

    os.makedirs(demo_input_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Create subsets (Breaths are 80 steps long)
    # Train: 50 breaths, Val: 10 breaths, Test: 10 breaths
    try:
        train_subset = pd.read_csv("./metadata/train.csv", nrows=50 * 80)
        val_subset = pd.read_csv("./metadata/validation.csv", nrows=10 * 80)
        test_subset = pd.read_csv("./metadata/test.csv", nrows=10 * 80)
    except FileNotFoundError as e:
        print(f"Error reading metadata: {e}")
        sys.exit(1)

    train_path = os.path.join(demo_input_dir, "train.csv")
    val_path = os.path.join(demo_input_dir, "val.csv")
    test_path = os.path.join(demo_input_dir, "test.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(
        f"Subsets created: Train({len(train_subset)}), Val({len(val_subset)}), Test({len(test_subset)})"
    )

    return train_path, val_path, test_path, demo_working_dir, demo_submission_dir


def override_config(train_path, val_path, test_path, working_dir, submission_dir):
    """
    Dynamically overrides Config attributes for the demo run.
    """
    print("--- Overriding Configuration ---")

    # Paths
    Config.TRAIN_PATH = train_path
    Config.VAL_PATH = val_path
    Config.TEST_PATH = test_path
    Config.WORKING_DIR = working_dir
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Training Hyperparameters (Optimized for speed)
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch for small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Model Architecture (Reduced capacity for speed)
    Config.CNN_FILTERS = 16
    Config.CNN_LAYERS = 2
    Config.LSTM_HIDDEN_SIZE = 32
    Config.LSTM_LAYERS = 1
    Config.DENSE_HIDDEN_SIZE = 32

    # Force CPU if GPU not available (though environment likely has GPU)
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Config updated: Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )


def verify_data_loading():
    """
    Verifies data processing and loader generation.
    """
    print("\n--- Verifying Data Loading ---")

    # Force reload to ensure we use the subset data
    train_loader, val_loader, test_loader = get_transformed_data(load_cached_data=False)

    # Check Train Loader
    x, y = next(iter(train_loader))
    print(f"Train Batch X: {x.shape}, Y: {y.shape}")

    # Assertions
    expected_seq_len = 80
    assert x.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        Config.INPUT_DIM,
    ), f"Expected X shape ({Config.BATCH_SIZE}, {expected_seq_len}, {Config.INPUT_DIM}), got {x.shape}"
    assert y.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
    ), f"Expected Y shape ({Config.BATCH_SIZE}, {expected_seq_len}), got {y.shape}"
    assert not torch.isnan(x).any(), "Input data contains NaNs"

    print("Data loading verification passed.")
    return x, y


def verify_model_logic(sample_x):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model Logic ---")

    model = DKRHNet().to(Config.DEVICE)
    sample_x = sample_x.to(Config.DEVICE)

    # Forward pass
    output = model(sample_x)
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 80), got {output.shape}"

    print("Model verification passed.")
    return model, output


def verify_loss_logic(model_out, target, input_tensor):
    """
    Verifies the custom masked loss function.
    """
    print("\n--- Verifying Loss Logic ---")

    criterion = MaskedL1Loss()
    target = target.to(Config.DEVICE)
    input_tensor = input_tensor.to(Config.DEVICE)

    loss = criterion(model_out, target, input_tensor)
    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert isinstance(loss.item(), float), "Loss should be a float"
    assert loss.item() >= 0, "Loss must be non-negative"

    print("Loss verification passed.")


def run_integration_test():
    """
    Runs the full training pipeline using the library's driver function.
    """
    print("\n--- Running Full Integration Test (Train/Val/Predict) ---")

    # This function handles the loop, saving, and submission generation
    train_model()

    # Verify submission file exists and has correct format
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Expected rows: 10 test breaths * 80 time steps = 800 rows
    expected_rows = 10 * 80
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    print("Integration test passed successfully.")


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        train_p, val_p, test_p, work_dir, sub_dir = setup_demo_environment()

        # 2. Configure
        override_config(train_p, val_p, test_p, work_dir, sub_dir)

        # 3. Verify Components
        sample_x, sample_y = verify_data_loading()
        model, sample_out = verify_model_logic(sample_x)
        verify_loss_logic(sample_out, sample_y, sample_x)

        # 4. Run Full Pipeline
        run_integration_test()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as ae:
        print(f"\nASSERTION FAILED: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Print traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
