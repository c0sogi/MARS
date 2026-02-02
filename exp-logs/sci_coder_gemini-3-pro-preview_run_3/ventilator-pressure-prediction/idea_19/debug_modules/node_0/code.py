import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_data_loaders, prepare_data
from library.model import WSDHNet
from library.train import run_training
from library.predict import generate_predictions


def create_subset_data(source_dir, dest_dir, n_breaths=20):
    """
    Creates small subsets of the data for demonstration purposes.
    Each breath has 80 time steps.
    """
    print(f"Creating data subsets (n_breaths={n_breaths}) in {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    # Calculate rows needed (80 steps per breath)
    n_rows = n_breaths * 80

    # 1. Train Subset
    train_df = pd.read_csv(os.path.join(source_dir, "train.csv"), nrows=n_rows)
    train_dst = os.path.join(dest_dir, "train.csv")
    train_df.to_csv(train_dst, index=False)

    # 2. Validation Subset
    val_df = pd.read_csv(os.path.join(source_dir, "validation.csv"), nrows=n_rows)
    val_dst = os.path.join(dest_dir, "validation.csv")
    val_df.to_csv(val_dst, index=False)

    # 3. Test Subset
    test_df = pd.read_csv(os.path.join(source_dir, "test.csv"), nrows=n_rows)
    test_dst = os.path.join(dest_dir, "test.csv")
    test_df.to_csv(test_dst, index=False)

    # 4. Sample Submission Subset
    # We need a sample submission that matches the test subset IDs
    sample_sub = pd.DataFrame({"id": test_df["id"], "pressure": 0})
    sub_dst = os.path.join(dest_dir, "sample_submission.csv")
    sample_sub.to_csv(sub_dst, index=False)

    return train_dst, val_dst, test_dst, sub_dst


def configure_demo_settings(demo_dir, train_path, val_path, test_path, sub_path):
    """
    Overrides the global Config class to use demo paths and lightweight hyperparameters.
    """
    print("Overriding Config settings for demonstration...")

    # Paths
    Config.EXP_ID = "demo_run"
    Config.WORKING_DIR = os.path.join(demo_dir, "working")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path
    Config.SAMPLE_SUBMISSION = sub_path

    # Cache paths (to avoid conflict with real run)
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test.parquet")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")

    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch for small data
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Model Architecture (Lightweight)
    Config.TCN_CHANNELS = [16, 32]
    Config.LSTM_HIDDEN_SIZE = 32
    Config.LSTM_LAYERS = 1
    Config.FUSION_HIDDEN_DIM = 64

    # Re-initialize to create directories
    Config.initialize()


def verify_data_pipeline():
    """
    Verifies that data loading works and produces correct shapes.
    """
    print("\n--- Verifying Data Pipeline ---")
    # Force processing from scratch
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Check Train Loader
    x, u_out, y = next(iter(train_loader))
    print(f"Train Batch Shape - X: {x.shape}, u_out: {u_out.shape}, y: {y.shape}")

    # Assertions
    # Shape should be [Batch, 80, Features]
    assert x.ndim == 3, "Input X should be 3-dimensional"
    assert x.shape[1] == 80, "Sequence length must be 80"
    assert u_out.shape == (Config.BATCH_SIZE, 80), "u_out mask shape mismatch"
    assert y.shape == (Config.BATCH_SIZE, 80), "Target y shape mismatch"

    print("Data Pipeline Verification Passed.")
    return x.shape[2], train_loader, val_loader, test_loader


def verify_model_architecture(input_dim, device):
    """
    Verifies that the model instantiates and performs a forward pass.
    """
    print("\n--- Verifying Model Architecture ---")
    model = WSDHNet(input_dim=input_dim).to(device)

    # Create dummy input
    dummy_x = torch.randn(2, 80, input_dim).to(device)

    # Forward pass
    output = model(dummy_x)
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 80), "Model output should be [Batch, Sequence_Length]"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model Architecture Verification Passed.")
    return model


def run_demo_training():
    """
    Runs the actual training loop using the library function.
    """
    print("\n--- Running Training Loop ---")
    # run_training handles initialization, data loading, training, and prediction
    # We pass load_cached_data=True because we just generated the cache in verify_data_pipeline
    # But to be safe and demonstrate full flow, we can let it reload or reuse.
    # Since we already called get_data_loaders(load_cached_data=False) above, cache exists.
    run_training(load_cached_data=True)

    # Verify Artifacts
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print(f"Model saved to {Config.MODEL_PATH}")


def verify_submission():
    """
    Verifies the generated submission file.
    """
    print("\n--- Verifying Submission ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        # If run_training didn't generate it (e.g. if we want to test predict separately)
        generate_predictions(load_cached_data=True)

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {sub_df.shape}")
    print(sub_df.head())

    # Load test subset to compare lengths
    test_df = pd.read_csv(Config.TEST_CSV)

    assert len(sub_df) == len(
        test_df
    ), f"Submission length {len(sub_df)} != Test length {len(test_df)}"
    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission columns missing"
    assert sub_df["pressure"].notna().all(), "Submission contains NaNs"

    print("Submission Verification Passed.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Setup
    seed_everything(42)
    device = get_device()

    # Define directories
    BASE_DIR = "./working/demo_execution"
    INPUT_META_DIR = "./metadata"

    # 2. Create Data Subsets
    train_path, val_path, test_path, sub_path = create_subset_data(
        INPUT_META_DIR,
        os.path.join(BASE_DIR, "input"),
        n_breaths=50,  # Use 50 breaths for a meaningful tiny run
    )

    # 3. Configure
    configure_demo_settings(BASE_DIR, train_path, val_path, test_path, sub_path)

    # 4. Verify Components
    input_dim, _, _, _ = verify_data_pipeline()
    verify_model_architecture(input_dim, device)

    # 5. Run Full Pipeline
    run_demo_training()

    # 6. Verify Results
    verify_submission()

    print("\n=== Demonstration Completed Successfully ===")
