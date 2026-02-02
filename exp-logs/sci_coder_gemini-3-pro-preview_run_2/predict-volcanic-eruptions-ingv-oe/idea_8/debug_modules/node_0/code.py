import os
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.feature_engineering import FeatureEngineer
from library.data_loader import get_dataloaders
from library.model import SeismicHybridModel
from library.engine import run_training, predict_fn


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # --------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Set DEBUG to True to use a small subset of data (20 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4  # Small batch size to ensure we have batches with 20 samples
    Config.NUM_WORKERS = 2

    # Use a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"

    # Update derived paths in Config to point to the new working directory
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.TARGET_MEAN_PATH = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    Config.TARGET_STD_PATH = os.path.join(Config.WORKING_DIR, "target_std.npy")

    # Ensure the directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Verify Utilities
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Utilities...")
    seed_everything(Config.SEED)

    # Test TargetScaler
    scaler = TargetScaler()
    dummy_targets = np.array([10.0, 20.0, 30.0])
    scaler.fit(dummy_targets)

    # Check stats
    assert scaler.mean == 20.0, f"Expected mean 20.0, got {scaler.mean}"
    assert np.isclose(scaler.std, 8.1649658), f"Unexpected std: {scaler.std}"

    # Check transform
    transformed = scaler.transform(np.array([20.0]))
    assert np.isclose(transformed[0], 0.0), "Transform of mean should be 0"

    # Check inverse transform
    inversed = scaler.inverse_transform(transformed)
    assert np.isclose(inversed[0], 20.0), "Inverse transform failed"
    print("TargetScaler verified.")

    # --------------------------------------------------------------------------
    # 3. Verify Feature Engineering
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Feature Engineering...")
    fe = FeatureEngineer()

    # Create a dummy signal: (Time, Channels)
    # Config.SIGNAL_LENGTH = 60001, Config.NUM_SENSORS = 10
    dummy_signal = np.random.randn(Config.SIGNAL_LENGTH, Config.NUM_SENSORS).astype(
        np.float32
    )

    # Test Spectrogram Generation
    spec = fe.get_spectrogram(dummy_signal)
    # Expected shape: [Channels, N_MELS, Time]
    # Time dimension depends on hop length. 60001 // 256 approx 234 + padding -> approx 235
    print(f"Spectrogram shape: {spec.shape}")
    assert spec.shape[0] == Config.NUM_SENSORS
    assert spec.shape[1] == Config.N_MELS
    assert isinstance(spec, torch.Tensor)

    # Test Time Features
    time_feats = fe.get_time_features(dummy_signal)
    assert "sensor_1_mean" in time_feats
    assert "sensor_10_kurt" in time_feats

    # Test Freq Features
    freq_feats = fe.get_freq_features(dummy_signal)
    assert "sensor_1_spec_centroid" in freq_feats
    print("Feature Engineering verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Data Loading
    # --------------------------------------------------------------------------
    print("\n>>> Initializing Data Loaders (this involves feature extraction)...")
    # Force reload to ensure pipeline runs
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=False
    )

    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")

    b_spec = batch["spectrogram"]
    b_tab = batch["tabular"]
    b_target = batch["target"]

    print(f"Batch Spectrogram Shape: {b_spec.shape}")
    print(f"Batch Tabular Shape: {b_tab.shape}")
    print(f"Batch Target Shape: {b_target.shape}")

    assert b_spec.shape[0] == Config.BATCH_SIZE
    assert b_spec.shape[1] == Config.NUM_SENSORS
    assert b_tab.shape[0] == Config.BATCH_SIZE
    assert b_target.shape[0] == Config.BATCH_SIZE
    print("Data Loading verified.")

    # --------------------------------------------------------------------------
    # 5. Verify Model
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Model...")
    num_tabular_features = b_tab.shape[1]
    model = SeismicHybridModel(num_tabular_features=num_tabular_features)
    model.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(b_spec.to(device), b_tab.to(device))

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1)
    print("Model architecture verified.")

    # --------------------------------------------------------------------------
    # 6. Verify Training Engine
    # --------------------------------------------------------------------------
    print("\n>>> Running Training Loop (2 Epochs)...")
    best_loss = run_training(model, train_loader, val_loader, device, target_scaler)

    assert isinstance(best_loss, float)
    assert best_loss < float("inf")
    print(f"Training finished. Best Loss: {best_loss}")

    # --------------------------------------------------------------------------
    # 7. Verify Prediction
    # --------------------------------------------------------------------------
    print("\n>>> Running Prediction on Test Set...")
    segment_ids, predictions = predict_fn(test_loader, model, device)

    print(f"Predictions Shape: {predictions.shape}")
    assert len(segment_ids) == len(predictions)
    assert len(segment_ids) > 0

    # Create submission dataframe example
    sub_df = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})
    print("Sample Prediction:")
    print(sub_df.head())

    # Inverse transform predictions to get real time
    real_predictions = target_scaler.inverse_transform(predictions)
    print(f"First prediction (real scale): {real_predictions[0]}")

    print("Prediction verified.")

    # --------------------------------------------------------------------------
    # 8. Cleanup
    # --------------------------------------------------------------------------
    print("\n>>> Cleaning up...")
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    print("Cleanup complete.")

    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demo()
