import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

# Import from the provided library
from library.config import Config
from library.data_processing import get_data_loaders, process_test_data
from library.model import CA_WRN
from library.trainer import run_training
from library.inference import run_inference


def setup_demo_config():
    """
    Overrides default Config parameters to run a fast demonstration.
    """
    print("Setting up demo configuration...")

    # Define demo-specific directories
    demo_working_dir = "./working/demo_run/working"
    demo_submission_dir = "./working/demo_run/submission"

    # Ensure clean slate
    if os.path.exists("./working/demo_run"):
        shutil.rmtree("./working/demo_run")
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Monkey-patch the Config class
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_submission_dir

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Small sample for speed

    # Reduce training parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.THRESHOLD_SEARCH_STEPS = 10  # Faster threshold search

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")


def verify_data_processing():
    """
    Demonstrates and verifies the data loading and processing pipeline.
    """
    print("\n=== Verifying Data Processing ===")

    # 1. Generate Data Loaders (forces processing from scratch via load_cached_data=False)
    train_loader, val_loader, center_indices, scaler = get_data_loaders(
        load_cached_data=False
    )

    # Assertions
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(center_indices) > 0, "No center indices identified for skip connection."
    assert scaler is not None, "Scaler was not fitted."

    # Check batch structure
    features, labels = next(iter(train_loader))
    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    assert features.dim() == 2, "Features should be 2D (Batch, InputDim)"
    assert labels.dim() == 1, "Labels should be 1D (Batch)"
    assert (
        features.shape[0] == Config.BATCH_SIZE
        or features.shape[0] <= Config.DEBUG_SAMPLE_SIZE
    )

    # Check cache creation
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "train_features.parquet")
    ), "Train cache missing."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "scaler.joblib")
    ), "Scaler cache missing."

    print("Data processing verification passed.")
    return features.shape[1], center_indices


def verify_model_architecture(input_dim, center_indices):
    """
    Demonstrates model instantiation and verifies the forward pass.
    """
    print("\n=== Verifying Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple verification

    # Instantiate Model
    model = CA_WRN(
        input_dim=input_dim,
        center_indices=center_indices,
        hidden_size=64,  # Smaller hidden size for demo
        num_layers=2,
        dropout=0.1,
    ).to(device)

    print(f"Model instantiated: CA_WRN with input_dim={input_dim}")

    # Create dummy input
    dummy_input = torch.randn(Config.BATCH_SIZE, input_dim).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs."

    print("Model architecture verification passed.")


def verify_training_pipeline():
    """
    Runs the training loop using the library's run_training function.
    """
    print("\n=== Verifying Training Pipeline ===")

    # run_training uses Config settings we patched earlier
    run_training()

    # Verify artifacts
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), f"Best model file not found at {model_path}"

    print("Training pipeline verification passed.")


def verify_inference_pipeline():
    """
    Runs the inference loop using the library's run_inference function.
    """
    print("\n=== Verifying Inference Pipeline ===")

    # run_inference loads the best model and generates submission
    run_inference()

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Load and check content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    assert "contact_id" in df_sub.columns, "Missing contact_id column."
    assert "contact" in df_sub.columns, "Missing contact column."
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary (0 or 1)."

    print("Inference pipeline verification passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Data Processing
    # We capture input_dim and center_indices here to verify the model next
    input_dim, center_indices = verify_data_processing()

    # 3. Model Logic
    verify_model_architecture(input_dim, center_indices)

    # 4. Training Loop
    # This will use the cached data generated in step 2
    verify_training_pipeline()

    # 5. Inference Loop
    # This will use the model saved in step 4
    verify_inference_pipeline()

    print("\nAll demonstrations completed successfully.")
