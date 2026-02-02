import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.model import DualResUNet
from library.trainer import Trainer, DecimatedMAELoss
from library.utils import set_seed


def create_mini_metadata():
    """
    Creates smaller metadata files for a quick demonstration run.
    Selects a single drive for train, val, and test to minimize processing time.
    """
    print("Creating mini metadata for demonstration...")

    # Create working directory for this demo
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select multiple drives for training to ensure valid data
    train_drives = train_meta["drive_id"].unique()
    if len(train_drives) > 0:
        # Select up to 3 drives to avoid empty dataset if one is bad
        selected_drives = train_drives[:3]
        mini_train = train_meta[train_meta["drive_id"].isin(selected_drives)].copy()
    else:
        mini_train = train_meta.head(100).copy()  # Fallback

    # Select one drive for validation
    val_drives = val_meta["drive_id"].unique()
    if len(val_drives) > 0:
        mini_val = val_meta[val_meta["drive_id"] == val_drives[0]].copy()
    else:
        mini_val = val_meta.head(100).copy()

    # Select one trip for testing
    test_trips = test_meta["tripId"].unique()
    if len(test_trips) > 0:
        mini_test = test_meta[test_meta["tripId"] == test_trips[0]].copy()
    else:
        mini_test = test_meta.head(100).copy()

    # Save mini metadata
    mini_train_path = os.path.join(demo_dir, "mini_train_meta.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val_meta.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test_meta.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini metadata saved to {demo_dir}")
    return mini_train_path, mini_val_path, mini_test_path, demo_dir


def validate_model_logic():
    """
    Validates the DualResUNet model architecture and DecimatedMAELoss.
    """
    print("\n--- Validating Model Architecture ---")

    # Instantiate model
    model = DualResUNet()
    model.eval()

    # Create dummy input tensors (Batch Size, Channels, Sequence Length)
    # Config defines IN_CHANNELS_A=10, IN_CHANNELS_B=10
    batch_size = 2
    seq_len = 128
    x_a = torch.randn(batch_size, Config.IN_CHANNELS_A, seq_len)
    x_b = torch.randn(batch_size, Config.IN_CHANNELS_B, seq_len)

    # Forward pass
    with torch.no_grad():
        output = model(x_a, x_b)

    # Check output shape: (Batch, NUM_CLASSES, Seq_Len)
    expected_shape = (batch_size, Config.NUM_CLASSES, seq_len)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model forward pass successful. Output shape:", output.shape)

    # Validate Loss Function
    print("\n--- Validating Loss Function ---")
    criterion = DecimatedMAELoss(aux_weight=0.4, decimation_factor=4)

    # Create dummy predictions (dict format for training mode)
    # Main output: Full resolution
    pred_main = torch.randn(batch_size, Config.NUM_CLASSES, seq_len, requires_grad=True)
    # Aux output: Decimated resolution (Seq_Len / 4)
    aux_len = seq_len // 4
    pred_aux = torch.randn(batch_size, Config.NUM_CLASSES, aux_len, requires_grad=True)

    preds = {"output": pred_main, "aux": pred_aux}

    # Targets are full resolution
    targets = torch.randn(batch_size, Config.NUM_CLASSES, seq_len)

    # Calculate loss
    loss = criterion(preds, targets)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Backprop check
    loss.backward()

    print(f"Loss calculation successful. Loss value: {loss.item():.4f}")


def run_demo_pipeline():
    """
    Runs the full training and inference pipeline using the Trainer class
    on the mini dataset.
    """
    print("\n--- Running Demo Pipeline ---")

    # 1. Setup Configuration for Demo
    # Override Config class attributes to use mini data and run fast
    mini_train_path, mini_val_path, mini_test_path, demo_dir = create_mini_metadata()

    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_PATH = os.path.join(demo_dir, "models", "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 2. Instantiate Trainer
    trainer = Trainer(run_dir=demo_dir)

    # 3. Load Data
    # load_cached=False forces processing of the new mini metadata
    print("Loading and processing data...")
    train_df, val_df, test_df = trainer.load_data(load_cached=False)

    print(f"Processed Train Data Shape: {train_df.shape}")
    print(f"Processed Val Data Shape: {val_df.shape}")
    print(f"Processed Test Data Shape: {test_df.shape}")

    # Assert data is not empty
    assert not train_df.empty, "Training dataframe is empty!"
    assert not val_df.empty, "Validation dataframe is empty!"
    assert not test_df.empty, "Test dataframe is empty!"

    # 4. Train
    print("Starting training...")
    trainer.fit(train_df, val_df)

    # Verify model file created
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved!"
    print("Training finished. Model saved.")

    # 5. Inference
    print("Generating submission...")
    trainer.generate_submission(test_df, Config.SUBMISSION_PATH)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Basic check on submission format
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in sub_df.columns, f"Submission missing column: {col}"

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # 1. Validate components
    validate_model_logic()

    # 2. Run end-to-end pipeline with mini data
    run_demo_pipeline()
