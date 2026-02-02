import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_processing import process_dataset
from library.dataset import GnssDataset, get_dataloaders
from library.model import ResUNet1D
from library.loss import DecimatedMAELoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(original_path, new_path, n_drives=1):
    """Creates a smaller metadata file containing only a few drives."""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original metadata not found at {original_path}")

    df = pd.read_csv(original_path)

    if "drive_id" in df.columns:
        # Sample n_drives
        drives = df["drive_id"].unique()[:n_drives]
        mini_df = df[df["drive_id"].isin(drives)].copy()
    else:
        # Fallback if drive_id not present (unlikely based on schema)
        mini_df = df.head(100).copy()

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    mini_df.to_csv(new_path, index=False)
    print(f"Created mini metadata at {new_path} with {len(mini_df)} rows.")
    return mini_df


def run_demo():
    print("=== Starting GNSS Pipeline Demo ===")

    # 1. Setup Configuration for Speed
    # We override the global Config class attributes to use a temporary workspace
    # and limit training duration.
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    Config.MODEL_PATH = os.path.join(demo_working_dir, "models", "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories based on new config
    Config.setup()
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    # 2. Create Mini Datasets
    # We point the config metadata paths to our new mini files
    print("\n[1] Creating Mini Metadata...")
    Config.TRAIN_METADATA_PATH = os.path.join(demo_working_dir, "mini_train_meta.csv")
    Config.VAL_METADATA_PATH = os.path.join(demo_working_dir, "mini_val_meta.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_working_dir, "mini_test_meta.csv")

    # Use the existing metadata in ./metadata to create subsets
    # We assume ./metadata/train_metadata.csv etc. exist as per problem description
    original_train_meta = "./metadata/train_metadata.csv"
    original_val_meta = "./metadata/val_metadata.csv"
    original_test_meta = "./metadata/test_metadata.csv"

    # Create subsets (1 drive each)
    create_mini_metadata(original_train_meta, Config.TRAIN_METADATA_PATH, n_drives=1)
    # Use train data for validation in demo to ensure we have data even if val split is empty/different
    create_mini_metadata(original_train_meta, Config.VAL_METADATA_PATH, n_drives=1)
    create_mini_metadata(original_test_meta, Config.TEST_METADATA_PATH, n_drives=1)

    # 3. Test Data Processing
    print("\n[2] Testing Data Processing...")
    # This function reads the GNSS logs pointed to by the metadata and aggregates features
    train_df = process_dataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    # Verification
    assert not train_df.empty, "Processed training dataframe is empty!"
    assert "UnixTimeMillis" in train_df.columns, "Missing timestamp column"
    assert "Target_E" in train_df.columns, "Missing Target East column"
    # Check for feature columns (L1/L5 stats)
    feature_cols_present = any(
        c.startswith("L1_") or c.startswith("L5_") for c in train_df.columns
    )
    assert feature_cols_present, "No L1/L5 feature columns found"
    print(f"Data processing successful. Shape: {train_df.shape}")

    # 4. Test Dataset and DataLoader
    print("\n[3] Testing Dataset and DataLoader...")
    # Initialize dataset
    ds = GnssDataset(Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True)
    assert len(ds) > 0, "Dataset has no trips"

    # Get a single item
    features, targets, meta = ds[0]
    print(f"Single item shapes - Features: {features.shape}, Targets: {targets.shape}")
    assert (
        features.shape[1] == Config.IN_CHANNELS
    ), f"Feature dim mismatch. Expected {Config.IN_CHANNELS}, got {features.shape[1]}"
    assert targets.shape[1] == 2, "Target dim mismatch. Expected 2 (E, N)"

    # Test DataLoader (Collate function)
    loader = get_dataloaders(debug=True)[0]  # Get train loader
    batch_features, batch_targets, batch_mask, batch_metas = next(iter(loader))

    print(
        f"Batch shapes - Features: {batch_features.shape}, Targets: {batch_targets.shape}, Mask: {batch_mask.shape}"
    )
    assert batch_features.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_features.shape[2] == Config.IN_CHANNELS, "Batch feature dim mismatch"

    # 5. Test Model and Loss
    print("\n[4] Testing Model and Loss...")
    device = Config.DEVICE
    model = ResUNet1D().to(device)
    criterion = DecimatedMAELoss().to(device)

    # Move batch to device and permute features for Conv1d (Batch, Channels, Length)
    inputs = batch_features.to(device).permute(0, 2, 1)
    labels = batch_targets.to(device)
    mask = batch_mask.to(device)

    # Forward pass
    outputs = model(inputs)
    # Outputs is a list: [final, aux3, aux2]
    assert len(outputs) == 3, "Model should return 3 outputs (Final + 2 Aux)"
    print(
        f"Model output shapes: Final {outputs[0].shape}, Aux1 {outputs[1].shape}, Aux2 {outputs[2].shape}"
    )

    # Check output shape matches input length (for final output)
    assert outputs[0].shape[2] == inputs.shape[2], "Output length mismatch"

    # Compute Loss
    loss = criterion(outputs, labels, mask)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 6. Test Trainer (Fit and Predict)
    print("\n[5] Testing Trainer (Fit & Predict)...")
    trainer = Trainer(debug=True)

    # Run training (1 epoch as configured)
    trainer.fit()

    # Check if model was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint not found after training"

    # Run inference
    trainer.predict()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")
    assert len(sub_df) > 0, "Submission file is empty"
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
