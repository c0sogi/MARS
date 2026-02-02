import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.feature_extractor import extract_features
from library.data_loader import prepare_data
from library.model import VolcanoMLP
from library.trainer import run_training
from library.inference import generate_predictions


def run_demo():
    print("=== Starting Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Runtime Configuration Override
    # ---------------------------------------------------------
    print("1. Configuring environment for demo...")

    # Define a specific working directory for this demo
    DEMO_DIR = "./working/demo"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.SEED = 123
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 files for speed
    Config.NUM_WORKERS = 0  # Use main process for simple debugging

    # Update derived paths in Config to point to the demo directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SCALER_SAVE_PATH = os.path.join(Config.WORKING_DIR, "scaler.npy")
    Config.TRAIN_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_CACHE = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print("   Configuration updated successfully.\n")

    # ---------------------------------------------------------
    # 2. Feature Extraction Demo
    # ---------------------------------------------------------
    print("2. Testing Feature Extraction...")

    # Extract features for a small subset
    # We force load_cached_data=False to ensure the extraction logic runs
    df_features = extract_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_FEATURES_CACHE,
        load_cached_data=False,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Validations
    assert isinstance(df_features, pd.DataFrame), "Output should be a pandas DataFrame"
    assert (
        len(df_features) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(df_features)}"
    assert "segment_id" in df_features.columns, "segment_id column missing"
    assert "time_to_eruption" in df_features.columns, "Target column missing"

    # Check for sensor feature columns (e.g., sensor_1_mean)
    sensor_cols = [c for c in df_features.columns if "sensor_" in c]
    assert len(sensor_cols) > 0, "No sensor features extracted"

    print(
        f"   Extracted {df_features.shape[1]} columns for {len(df_features)} segments."
    )
    print("   Feature extraction verified.\n")

    # ---------------------------------------------------------
    # 3. Data Loader Demo
    # ---------------------------------------------------------
    print("3. Testing Data Loading Pipeline...")

    # prepare_data handles scaling and splitting
    train_loader, val_loader, scaler, input_dim = prepare_data(
        debug_size=Config.DEBUG_SAMPLE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,  # Use the cache we just created
    )

    # Validations
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert input_dim > 0, "Input dimension must be positive"

    # Fetch one batch to verify shapes
    X_batch, y_batch = next(iter(train_loader))

    assert X_batch.shape[0] <= Config.BATCH_SIZE, "Batch size mismatch"
    assert (
        X_batch.shape[1] == input_dim
    ), f"Feature dimension mismatch. Expected {input_dim}, got {X_batch.shape[1]}"
    assert y_batch.ndim == 1, "Target should be a 1D tensor"

    # Check scaler files creation
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "scaler_mean.npy")
    ), "Scaler mean file not saved"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "scaler_scale.npy")
    ), "Scaler scale file not saved"

    print(f"   Input Dimension: {input_dim}")
    print(f"   Batch Shape: {X_batch.shape}")
    print("   Data loading verified.\n")

    # ---------------------------------------------------------
    # 4. Model Architecture Demo
    # ---------------------------------------------------------
    print("4. Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = VolcanoMLP(
        input_dim=input_dim, hidden_layers=[32, 16], dropout_rate=0.1
    ).to(device)

    # Forward pass check
    with torch.no_grad():
        output = model(X_batch.to(device))

    assert output.shape == (
        X_batch.shape[0],
        1,
    ), f"Output shape mismatch. Expected {(X_batch.shape[0], 1)}, got {output.shape}"

    print("   Model instantiation and forward pass successful.\n")

    # ---------------------------------------------------------
    # 5. Training Pipeline Demo
    # ---------------------------------------------------------
    print("5. Running Training Loop (1 Epoch)...")

    # Run the full training orchestrator
    trained_model = run_training(
        debug_size=Config.DEBUG_SAMPLE_SIZE,
        epochs=Config.EPOCHS,
        lr=1e-3,
        patience=1,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Validations
    assert isinstance(trained_model, torch.nn.Module)
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    print("   Training loop completed and model saved.\n")

    # ---------------------------------------------------------
    # 6. Inference Pipeline Demo
    # ---------------------------------------------------------
    print("6. Running Inference Pipeline...")

    # Generate predictions on a subset of test data
    df_submission = generate_predictions(
        model=trained_model,
        device=device,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
        load_cached_data=False,  # Force process test data
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Validations
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    assert (
        len(df_submission) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_submission)}"
    assert list(df_submission.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns incorrect"
    assert not df_submission.isnull().values.any(), "Submission contains NaNs"

    print(f"   Submission generated at {Config.SUBMISSION_PATH}")
    print("   Inference pipeline verified.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
