import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import GNSSPreprocessor
from library.dataset import SKFDataset
from library.model import SKFNet
from library.trainer import Trainer
from library.inference import InferenceEngine


def setup_demo_environment():
    """
    Sets up a specific working directory for the demo and overrides Config paths
    to point to subsets of data and the new working directory.
    """
    demo_work_dir = "./working/demo_execution"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir)

    print(f"[Demo] Created working directory: {demo_work_dir}")

    # Override Config parameters for speed
    Config.WORK_DIR = demo_work_dir
    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.WINDOW_SIZE = 15  # Keep default or reduce if needed

    # Update paths in Config to point to the demo directory
    # Note: Since Config attributes are initialized at import time, we must update them manually.

    # Metadata paths (will point to subsets we create)
    Config.TRAIN_METADATA_PATH = os.path.join(
        demo_work_dir, "train_metadata_subset.csv"
    )
    Config.VAL_METADATA_PATH = os.path.join(demo_work_dir, "val_metadata_subset.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_work_dir, "test_metadata_subset.csv")

    # Cache paths
    Config.CACHE_TRAIN_X_SEQ = os.path.join(demo_work_dir, "train_X_kinematic.npy")
    Config.CACHE_TRAIN_X_SKY = os.path.join(demo_work_dir, "train_X_sky.npy")
    Config.CACHE_TRAIN_Y = os.path.join(demo_work_dir, "train_y.npy")
    Config.CACHE_TRAIN_META = os.path.join(demo_work_dir, "train_meta.parquet")

    Config.CACHE_VAL_X_SEQ = os.path.join(demo_work_dir, "val_X_kinematic.npy")
    Config.CACHE_VAL_X_SKY = os.path.join(demo_work_dir, "val_X_sky.npy")
    Config.CACHE_VAL_Y = os.path.join(demo_work_dir, "val_y.npy")
    Config.CACHE_VAL_META = os.path.join(demo_work_dir, "val_meta.parquet")

    Config.CACHE_TEST_X_SEQ = os.path.join(demo_work_dir, "test_X_kinematic.npy")
    Config.CACHE_TEST_X_SKY = os.path.join(demo_work_dir, "test_X_sky.npy")
    Config.CACHE_TEST_META = os.path.join(demo_work_dir, "test_meta.parquet")

    Config.SCALER_PATH = os.path.join(demo_work_dir, "scaler.json")
    Config.MODEL_PATH = os.path.join(demo_work_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    return demo_work_dir


def create_data_subsets():
    """
    Reads the original metadata files, samples a few trips, and saves them as subsets
    referenced by the updated Config paths.
    """
    print("[Demo] Creating data subsets...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/validation_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Select 1 trip for training
    train_trips = orig_train_meta["tripId"].unique()
    train_subset = orig_train_meta[orig_train_meta["tripId"] == train_trips[0]].copy()
    # Limit to first 500 rows to speed up feature engineering window loop
    train_subset = train_subset.iloc[:500]
    train_subset.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    print(
        f"[Demo] Saved train subset: {len(train_subset)} rows, Trip: {train_trips[0]}"
    )

    # Select 1 trip for validation
    val_trips = orig_val_meta["tripId"].unique()
    val_subset = orig_val_meta[orig_val_meta["tripId"] == val_trips[0]].copy()
    val_subset = val_subset.iloc[:200]
    val_subset.to_csv(Config.VAL_METADATA_PATH, index=False)
    print(f"[Demo] Saved val subset: {len(val_subset)} rows, Trip: {val_trips[0]}")

    # Select 1 trip for testing
    test_trips = orig_test_meta["tripId"].unique()
    test_subset = orig_test_meta[orig_test_meta["tripId"] == test_trips[0]].copy()
    test_subset = test_subset.iloc[:200]
    test_subset.to_csv(Config.TEST_METADATA_PATH, index=False)
    print(f"[Demo] Saved test subset: {len(test_subset)} rows, Trip: {test_trips[0]}")


def demo_preprocessing():
    """
    Demonstrates the GNSSPreprocessor class.
    """
    print("\n[Demo] --- Preprocessing Step ---")
    preprocessor = GNSSPreprocessor()

    # Process Train
    print("[Demo] Processing Train Data...")
    X_seq_train, X_sky_train, y_train, meta_train = preprocessor.process_train(
        load_cached_data=False
    )

    # Validation
    assert X_seq_train is not None, "Train Sequence data is None"
    assert len(X_seq_train) == len(y_train), "Mismatch in train data and labels"
    assert (
        X_seq_train.shape[1] == Config.WINDOW_SIZE
    ), f"Window size mismatch. Expected {Config.WINDOW_SIZE}, got {X_seq_train.shape[1]}"
    print(
        f"[Demo] Train Data Shapes: Seq={X_seq_train.shape}, Sky={X_sky_train.shape}, Y={y_train.shape}"
    )

    # Process Val
    print("[Demo] Processing Validation Data...")
    X_seq_val, X_sky_val, y_val, meta_val = preprocessor.process_val(
        load_cached_data=False
    )
    print(
        f"[Demo] Val Data Shapes: Seq={X_seq_val.shape}, Sky={X_sky_val.shape}, Y={y_val.shape}"
    )

    return X_seq_train, X_sky_train, y_train, X_seq_val, X_sky_val, y_val


def demo_dataset_and_model(X_seq, X_sky, y):
    """
    Demonstrates SKFDataset and SKFNet.
    """
    print("\n[Demo] --- Dataset and Model Step ---")

    # Dataset
    dataset = SKFDataset(X_seq, X_sky, y)
    sample_seq, sample_sky, sample_y = dataset[0]

    print(
        f"[Demo] Dataset Sample Shapes: Seq={sample_seq.shape}, Sky={sample_sky.shape}, Y={sample_y.shape}"
    )

    # Check shape for CNN input (Channels, Length)
    # Original X_seq is (N, Length, Channels), Dataset permutes to (Channels, Length)
    assert (
        sample_seq.shape[0] == X_seq.shape[2]
    ), "Dataset did not permute channels correctly for CNN"

    # Model
    model = SKFNet()
    # Create a batch
    batch_seq = sample_seq.unsqueeze(0)  # (1, Channels, Length)
    batch_sky = sample_sky.unsqueeze(0)  # (1, SkyFeatures)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_seq, batch_sky)

    print(f"[Demo] Model Output Shape: {output.shape}")
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("[Demo] Model forward pass successful.")


def demo_training():
    """
    Demonstrates the Trainer class.
    """
    print("\n[Demo] --- Training Step ---")

    # We need to reload data via preprocessor to ensure we pass the correct objects or load from cache
    # Since we just ran preprocessing and saved to cache, we can load from cache now.
    preprocessor = GNSSPreprocessor()

    # Load data (simulating a fresh run that uses cached data)
    X_seq_train, X_sky_train, y_train, _ = preprocessor.process_train(
        load_cached_data=True
    )
    X_seq_val, X_sky_val, y_val, _ = preprocessor.process_val(load_cached_data=True)

    train_dataset = SKFDataset(X_seq_train, X_sky_train, y_train)
    val_dataset = SKFDataset(X_seq_val, X_sky_val, y_val)

    trainer = Trainer()
    trainer.fit(train_dataset, val_dataset)

    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved after training"
    print(f"[Demo] Training finished. Model saved at {Config.MODEL_PATH}")


def demo_inference():
    """
    Demonstrates the InferenceEngine.
    """
    print("\n[Demo] --- Inference Step ---")

    engine = InferenceEngine()
    # Run inference. This will:
    # 1. Process test data (using the subset metadata we created)
    # 2. Load the model trained in the previous step
    # 3. Generate submission.csv
    engine.generate_submission(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"[Demo] Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())

    # Verify columns
    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Submission missing required columns"


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    setup_demo_environment()

    # 2. Create Data Subsets
    create_data_subsets()

    # 3. Preprocessing
    X_seq_train, X_sky_train, y_train, X_seq_val, X_sky_val, y_val = (
        demo_preprocessing()
    )

    # 4. Dataset & Model Check
    demo_dataset_and_model(X_seq_train, X_sky_train, y_train)

    # 5. Training Loop
    demo_training()

    # 6. Inference Loop
    demo_inference()

    print("\n[Demo] All demonstrations completed successfully.")
