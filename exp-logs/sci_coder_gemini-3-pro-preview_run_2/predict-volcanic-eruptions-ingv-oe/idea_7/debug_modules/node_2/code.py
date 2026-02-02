import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.feature_engineering import FeatureEngineer
from library.dataset import VolcanoDataset
from library.model import HybridModel
from library.train import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Library Usage Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Patch Config for speed, isolation, and CPU execution
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10  # Process only 10 samples
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
    Config.DEVICE = "cpu"  # Use CPU for simple functional verification

    # Define a separate demo working directory to avoid conflicts
    DEMO_ROOT = "./working/demo_execution"
    Config.WORKING_DIR = os.path.join(DEMO_ROOT, "working")
    Config.SUBMISSION_DIR = os.path.join(DEMO_ROOT, "submission")

    # Manually update derived paths in Config since they are static class attributes
    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update file paths
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.STATS_SCALER_MEAN_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_mean.npy"
    )
    Config.STATS_SCALER_SCALE_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_scale.npy"
    )
    Config.TARGET_MEAN_PATH = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    Config.TARGET_STD_PATH = os.path.join(Config.WORKING_DIR, "target_std.npy")

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration patched successfully.")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Utilities (TargetScaler)
    # -------------------------------------------------------------------------
    print("\n[2] Testing TargetScaler...")
    scaler = TargetScaler()
    dummy_targets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Fit
    scaler.fit(dummy_targets)
    assert scaler.mean == 30.0, f"Expected mean 30.0, got {scaler.mean}"
    assert np.isclose(scaler.std, 14.1421356), f"Unexpected std: {scaler.std}"

    # Transform
    scaled = scaler.transform(dummy_targets)
    expected_first = (10.0 - 30.0) / 14.1421356
    assert np.isclose(scaled[0], expected_first), "Scaling logic incorrect"

    # Inverse Transform
    reconstructed = scaler.inverse_transform(scaled)
    assert np.allclose(dummy_targets, reconstructed), "Inverse transform failed"

    # Save and Load
    mean_path = os.path.join(Config.WORKING_DIR, "test_mean.npy")
    std_path = os.path.join(Config.WORKING_DIR, "test_std.npy")
    scaler.save(mean_path, std_path)

    scaler_loaded = TargetScaler()
    scaler_loaded.load(mean_path, std_path)
    assert scaler_loaded.mean == scaler.mean
    print("TargetScaler verified.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[3] Running Feature Engineering (Debug Mode)...")
    # This will process the first 10 files from train, val, and test metadata
    # and save parquet files to our demo working directory
    fe = FeatureEngineer()
    fe.run(load_cached_data=False)

    # Verify artifacts
    assert os.path.exists(
        Config.TRAIN_FEATURES_PATH
    ), "Train features parquet not created"
    assert os.path.exists(Config.STATS_SCALER_MEAN_PATH), "Scaler stats not created"

    # Verify content
    df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    assert (
        len(df_train) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} rows, got {len(df_train)}"
    # Check for a specific feature column
    assert "sensor_1_mean" in df_train.columns, "Missing tabular features"
    print(f"Feature Engineering complete. Generated {len(df_train)} samples.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[4] Testing VolcanoDataset...")
    # Initialize dataset (will load the parquet generated above)
    train_dataset = VolcanoDataset(mode="train")

    # Check length
    assert len(train_dataset) == Config.DEBUG_SIZE

    # Fetch one sample
    spec, tabular, target, seg_id = train_dataset[0]

    # Check Spectrogram Shape: (Channels, Freq, Time)
    # Channels=10, N_MELS=128 (from Config)
    # Time depends on signal length (60001) and hop length (256) -> ~235
    print(f"Spectrogram shape: {spec.shape}")
    assert spec.dim() == 3
    assert spec.shape[0] == 10
    assert spec.shape[1] == 128
    # spec.shape[2] should be around 235

    # Check Tabular Shape
    print(f"Tabular features shape: {tabular.shape}")
    assert tabular.dim() == 1
    tabular_dim = tabular.shape[0]

    # Check Target
    print(f"Target value (scaled): {target.item()}")
    assert isinstance(target, torch.Tensor)

    print("Dataset verified.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Model
    # -------------------------------------------------------------------------
    print("\n[5] Testing HybridModel...")
    model = HybridModel(tabular_input_dim=tabular_dim)
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    # Expand single sample to batch
    dummy_spec = spec.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(Config.DEVICE)
    dummy_tabular = tabular.unsqueeze(0).repeat(batch_size, 1).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_spec, dummy_tabular)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {output.shape}"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Training Step
    # -------------------------------------------------------------------------
    print("\n[6] Testing Training Loop...")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Val dataset
    val_dataset = VolcanoDataset(mode="val")
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Train loss is NaN"

    # Validate
    print("Running validate...")
    val_loss, val_mae = validate(
        model, val_loader, criterion, Config.DEVICE, scaler=train_dataset.scaler
    )
    print(f"Val Loss: {val_loss:.4f}, Val MAE (Original): {val_mae:.4f}")
    assert not np.isnan(val_loss), "Val loss is NaN"

    print("Training loop verified.")
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
