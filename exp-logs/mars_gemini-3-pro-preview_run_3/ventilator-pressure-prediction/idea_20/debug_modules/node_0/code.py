import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import CWDHNet, MaskedMAELoss
from library.train import run_training
from library.inference import generate_predictions


def setup_demo_config():
    """
    Overrides the default configuration for a fast demonstration run.
    """
    print("1. Setting up demo configuration...")

    # Enable Debug mode to use a tiny subset of data (100 breaths)
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16

    # Set up a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Update dependent paths manually since they were defined at class level
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.SUBMISSION_FILE = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # We also need to ensure the cache files point to the new working dir
    # Note: The prepare_datasets function uses Config.WORKING_DIR to determine cache location,
    # so setting Config.WORKING_DIR is sufficient for the .npy files.

    Config.print_config()
    print("Configuration updated for demo.\n")


def verify_data_processing():
    """
    Demonstrates data loading and verifies shapes.
    """
    print("2. Verifying Data Processing...")

    # Force reload to generate cache in the new demo directory
    train_x, train_y, val_x, val_y, test_x = prepare_datasets(load_cached_data=False)

    # Verify shapes
    # Shape should be (N_breaths, 80, N_features)
    print(f"Train X Shape: {train_x.shape}")
    print(f"Train Y Shape: {train_y.shape}")

    assert train_x.ndim == 3, "Train X should be 3D"
    assert train_x.shape[1] == 80, "Sequence length must be 80"
    assert train_x.shape[2] == len(Config.CONT_FEATURES), "Feature dimension mismatch"
    assert train_y.shape == (train_x.shape[0], 80), "Target shape mismatch"

    # Check for NaNs
    assert not np.isnan(train_x).any(), "NaNs found in training data"

    print("Data processing verification passed.\n")
    return train_x, train_y


def verify_model_architecture():
    """
    Instantiates the model and runs a dummy forward pass.
    """
    print("3. Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = CWDHNet().to(device)
    model.eval()

    # Create dummy input: (Batch=4, Seq=80, Feat=Input_Dim)
    dummy_input = torch.randn(4, 80, Config.INPUT_DIM).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Output should be (Batch, Seq) -> (4, 80)
    assert output.shape == (4, 80), f"Expected output (4, 80), got {output.shape}"

    print("Model architecture verification passed.\n")


def verify_loss_function():
    """
    Verifies that MaskedMAELoss correctly ignores the expiratory phase.
    """
    print("4. Verifying Masked Loss Function...")

    criterion = MaskedMAELoss()

    # Create synthetic data
    # Batch=1, Seq=4
    preds = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    targets = torch.tensor([[12.0, 12.0, 50.0, 50.0]])  # Errors: 2, 2, 40, 40

    # Construct inputs to define u_out
    # We need to know where u_out is in the feature list
    u_out_idx = Config.CONT_FEATURES.index("u_out")

    # Create input tensor (Batch, Seq, Feat)
    inputs = torch.zeros(1, 4, len(Config.CONT_FEATURES))

    # Set u_out: 0, 0, 1, 1 (Inspiratory, Inspiratory, Expiratory, Expiratory)
    inputs[0, 0, u_out_idx] = 0
    inputs[0, 1, u_out_idx] = 0
    inputs[0, 2, u_out_idx] = 1
    inputs[0, 3, u_out_idx] = 1

    # Expected Loss:
    # Time 0: |10 - 12| = 2 (Mask=1)
    # Time 1: |10 - 12| = 2 (Mask=1)
    # Time 2: |10 - 50| = 40 (Mask=0) -> Ignored
    # Time 3: |10 - 50| = 40 (Mask=0) -> Ignored
    # Mean = (2 + 2) / 2 = 2.0

    loss = criterion(preds, targets, inputs)
    print(f"Calculated Loss: {loss.item()}")

    assert (
        abs(loss.item() - 2.0) < 1e-6
    ), f"Loss calculation incorrect. Expected 2.0, got {loss.item()}"

    print("Loss function verification passed.\n")


def run_pipeline_demo():
    """
    Runs the actual training and inference pipeline using library functions.
    """
    print("5. Running Training Pipeline...")

    # Run training (uses Config.EPOCHS=2, Config.DEBUG=True)
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # Check if model was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model not saved at {Config.MODEL_PATH}")
    print(f"Model saved successfully at {Config.MODEL_PATH}")

    print("\n6. Running Inference Pipeline...")

    # Generate predictions
    generate_predictions(batch_size=Config.BATCH_SIZE)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission loaded. Shape: {df.shape}")
    print(df.head())

    # In DEBUG mode, we take 50 breaths for test. 50 * 80 = 4000 rows.
    # Check if we have roughly the expected number of rows
    expected_rows = 50 * 80
    assert (
        len(df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df)}"
    assert (
        "id" in df.columns and "pressure" in df.columns
    ), "Missing columns in submission"

    print("Pipeline demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup
    setup_demo_config()

    # 2. Verify Data
    verify_data_processing()

    # 3. Verify Model
    verify_model_architecture()

    # 4. Verify Loss
    verify_loss_function()

    # 5. Run Pipeline (Train + Inference)
    run_pipeline_demo()
