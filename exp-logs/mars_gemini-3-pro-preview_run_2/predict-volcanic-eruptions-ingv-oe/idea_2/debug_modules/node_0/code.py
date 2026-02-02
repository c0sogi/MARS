import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.feature_engineering import SeismicFeatureEngineer
from library.dataset import get_dataloaders
from library.model import HybridModel
from library.train import train_one_epoch, validate
from library.predict import generate_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(original_path, mini_path, n_samples=20):
    """Creates a subset of the metadata for fast demonstration."""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original metadata not found: {original_path}")

    df = pd.read_csv(original_path)
    # Take top N samples to ensure files exist
    df_mini = df.head(n_samples).copy()

    os.makedirs(os.path.dirname(mini_path), exist_ok=True)
    df_mini.to_csv(mini_path, index=False)
    print(f"Created mini metadata at {mini_path} with {len(df_mini)} samples.")


def run_demo():
    print("=== Starting Seismic Prediction Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    print("--- 1. Configuration & Data Setup ---")

    # Define paths for the demo
    DEMO_DIR = "./working/demo_execution"
    MINI_META_DIR = os.path.join(DEMO_DIR, "metadata")

    # Create mini metadata files to speed up feature engineering
    # Original metadata is in ./metadata/
    create_mini_metadata(
        "./metadata/train.csv", os.path.join(MINI_META_DIR, "train.csv"), n_samples=20
    )
    create_mini_metadata(
        "./metadata/val.csv", os.path.join(MINI_META_DIR, "val.csv"), n_samples=10
    )
    create_mini_metadata(
        "./metadata/test.csv", os.path.join(MINI_META_DIR, "test.csv"), n_samples=10
    )

    # Monkey-patch Config to use demo paths and settings
    Config.WORKING_DIR = os.path.join(DEMO_DIR, "working")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Update Metadata Paths
    Config.TRAIN_METADATA = os.path.join(MINI_META_DIR, "train.csv")
    Config.VAL_METADATA = os.path.join(MINI_META_DIR, "val.csv")
    Config.TEST_METADATA = os.path.join(MINI_META_DIR, "test.csv")

    # Update Output Paths
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Update Cache Paths
    Config.TARGET_MEAN_PATH = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    Config.TARGET_STD_PATH = os.path.join(Config.WORKING_DIR, "target_std.npy")
    Config.STATS_SCALER_MEAN_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_mean.npy"
    )
    Config.STATS_SCALER_SCALE_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_scale.npy"
    )
    Config.SPEC_MEAN_PATH = os.path.join(Config.WORKING_DIR, "spec_mean.npy")
    Config.SPEC_STD_PATH = os.path.join(Config.WORKING_DIR, "spec_std.npy")
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    # Optimize Hyperparameters for Speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated for demo run.")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n--- 2. Verifying Utilities (TargetScaler) ---")
    scaler = TargetScaler()
    dummy_targets = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    scaler.fit(dummy_targets)

    transformed = scaler.transform(dummy_targets)
    inverse = scaler.inverse_transform(transformed)

    assert np.allclose(
        dummy_targets, inverse, atol=1e-5
    ), "TargetScaler inverse transform failed."
    print("TargetScaler logic verified.")

    # ---------------------------------------------------------
    # 3. Verify Feature Engineering
    # ---------------------------------------------------------
    print("\n--- 3. Verifying Feature Engineering ---")
    fe = SeismicFeatureEngineer()

    # Load one sample file to test computation
    sample_meta = pd.read_csv(Config.TRAIN_METADATA).iloc[0]
    sample_path = os.path.join(Config.INPUT_DIR, sample_meta["file_path"])
    df_sample = pd.read_csv(sample_path)

    # Test Spectrogram
    spec = fe.compute_spectrogram(df_sample)
    print(f"Spectrogram Shape: {spec.shape}")
    # Expected: (10 sensors, 64 mels, ~235 time steps)
    assert spec.shape[0] == 10 and spec.shape[1] == 64, "Spectrogram shape mismatch."

    # Test Statistics
    stats = fe.compute_statistics(df_sample)
    print(f"Computed {len(stats)} statistical features.")
    assert len(stats) > 0, "No statistics computed."

    # ---------------------------------------------------------
    # 4. Data Pipeline (Get DataLoaders)
    # ---------------------------------------------------------
    print("\n--- 4. Initializing Data Pipeline ---")
    # This will trigger caching of features and scalers
    loaders = get_dataloaders(load_cached_data=False)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Verify Batch
    spec_batch, tab_batch, target_batch = next(iter(train_loader))
    print(
        f"Train Batch - Spec: {spec_batch.shape}, Tab: {tab_batch.shape}, Target: {target_batch.shape}"
    )

    assert spec_batch.shape[0] == Config.BATCH_SIZE
    assert target_batch.shape[0] == Config.BATCH_SIZE

    num_tabular_features = tab_batch.shape[1]
    print(f"Tabular Feature Dimension: {num_tabular_features}")

    # ---------------------------------------------------------
    # 5. Model Initialization
    # ---------------------------------------------------------
    print("\n--- 5. Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridModel(num_tabular_features=num_tabular_features)
    model = model.to(device)

    # Test Forward Pass
    with torch.no_grad():
        dummy_out = model(spec_batch.to(device), tab_batch.to(device))

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (Config.BATCH_SIZE,), "Model output shape mismatch."

    # ---------------------------------------------------------
    # 6. Training Loop
    # ---------------------------------------------------------
    print("\n--- 6. Running Training Loop ---")
    # We implement the loop manually here to demonstrate usage of train_one_epoch
    # and to bypass the 'train_epoch' naming issue in the provided library/train.py

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Load scaler for validation metric
    target_scaler = TargetScaler()
    target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_mae = validate(
            val_loader, model, criterion, target_scaler, device
        )

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}"
        )

    # Save Model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")
    assert os.path.exists(Config.MODEL_PATH), "Model file was not created."

    # ---------------------------------------------------------
    # 7. Inference / Prediction
    # ---------------------------------------------------------
    print("\n--- 7. Running Inference ---")

    # Generate predictions using the library function
    # This uses the test loader and the saved model
    generate_predictions(load_cached_data=True, device=device)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    # Check format
    expected_cols = ["segment_id", "time_to_eruption"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert (
        len(df_sub) == 10
    ), "Submission row count mismatch (expected 10 from mini metadata)."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
