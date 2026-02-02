import os
import sys
import pandas as pd
import numpy as np
import torch

# ------------------------------------------------------------------------------
# 1. Configuration Setup
# ------------------------------------------------------------------------------
# We import the Config class and patch it to run a fast demonstration.
from library.config import Config

print("Setting up configuration for fast demonstration...")

# Enable Debug mode to use a tiny subset of data
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples for speed

# Reduce Training parameters
Config.EPOCHS = 1
Config.BATCH_SIZE = 16
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in this script

# Reduce Model complexity for faster initialization and forward pass
Config.HIDDEN_DIM = 64
Config.EMBEDDING_DIM = 8
Config.NUM_RES_BLOCKS = 1

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Metadata Handling
# ------------------------------------------------------------------------------
# The Trainer.generate_submission method reads the test metadata file to get IDs.
# Since we are running in DEBUG mode with only 100 samples, we must create
# truncated metadata files so the row counts match during submission generation.


def create_truncated_metadata(source_path, dest_name, n_rows):
    """Reads source metadata, takes top n_rows, saves to working dir."""
    df = pd.read_csv(source_path)
    df = df.head(n_rows)
    dest_path = os.path.join(Config.WORKING_DIR, dest_name)
    df.to_csv(dest_path, index=False)
    return dest_path


print("Creating truncated metadata files...")
Config.TRAIN_METADATA = create_truncated_metadata(
    os.path.join(Config.METADATA_DIR, "train_metadata.csv"),
    "train_meta_debug.csv",
    Config.DEBUG_SAMPLE_SIZE,
)
Config.VAL_METADATA = create_truncated_metadata(
    os.path.join(Config.METADATA_DIR, "val_metadata.csv"),
    "val_meta_debug.csv",
    Config.DEBUG_SAMPLE_SIZE,
)
Config.TEST_METADATA = create_truncated_metadata(
    os.path.join(Config.METADATA_DIR, "test_metadata.csv"),
    "test_meta_debug.csv",
    Config.DEBUG_SAMPLE_SIZE,
)

# ------------------------------------------------------------------------------
# 3. Import Library Modules
# ------------------------------------------------------------------------------
# Imports are done after config patching, though Python object references would
# handle it regardless.
from library import data_utils
from library import dataset
from library.model import ResMLP
from library.trainer import Trainer

# ------------------------------------------------------------------------------
# 4. Demonstration & Verification
# ------------------------------------------------------------------------------


def demo_data_processing():
    print("\n--- Demonstrating Data Processing ---")
    # Force processing from scratch (load_cached_data=False) to verify logic
    # This uses the truncated metadata we just set up.
    data = data_utils.process_data(load_cached_data=False)

    # Verify Keys
    expected_keys = [
        "X_num_train",
        "X_cat_train",
        "y_train",
        "X_num_val",
        "X_cat_val",
        "y_val",
        "X_num_test",
        "X_cat_test",
    ]
    for k in expected_keys:
        assert k in data, f"Missing key in processed data: {k}"

    # Verify Shapes
    # Should match DEBUG_SAMPLE_SIZE (100) and feature dimensions
    assert data["X_num_train"].shape == (
        Config.DEBUG_SAMPLE_SIZE,
        Config.NUM_CONTINUOUS_FEATURES,
    )
    assert data["X_cat_train"].shape == (
        Config.DEBUG_SAMPLE_SIZE,
        Config.F_27_SEQ_LENGTH,
    )
    assert data["y_train"].shape == (Config.DEBUG_SAMPLE_SIZE,)

    print("Data processing output verified.")


def demo_dataset():
    print("\n--- Demonstrating Dataset Class ---")
    train_ds, val_ds, test_ds = dataset.get_datasets(load_cached_data=True)

    # Verify Length
    assert len(train_ds) == Config.DEBUG_SAMPLE_SIZE

    # Verify Item Structure
    sample = train_ds[0]
    assert "continuous" in sample
    assert "categorical" in sample
    assert "target" in sample

    # Verify Tensors
    assert isinstance(sample["continuous"], torch.Tensor)
    assert sample["continuous"].dtype == torch.float32
    assert sample["categorical"].dtype == torch.long

    print("Dataset class verified.")


def demo_model():
    print("\n--- Demonstrating Model Architecture ---")
    model = ResMLP()
    model.eval()

    # Create dummy input batch
    batch_size = 4
    dummy_cont = torch.randn(batch_size, Config.NUM_CONTINUOUS_FEATURES)
    # Categorical indices must be within VOCAB_SIZE
    dummy_cat = torch.randint(
        0, Config.VOCAB_SIZE, (batch_size, Config.F_27_SEQ_LENGTH)
    )

    # Forward Pass
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Verify Output Shape: (Batch, 1)
    assert output.shape == (batch_size, 1)
    print("Model forward pass verified.")


def demo_training_and_submission():
    print("\n--- Demonstrating Training & Submission ---")
    trainer = Trainer()

    # 1. Train
    print("Running training (1 epoch)...")
    trainer.fit(epochs=Config.EPOCHS, patience=1, load_cached_data=True)

    # Verify Model Artifact
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."

    # 2. Generate Submission
    print("Generating submission...")
    trainer.generate_submission(load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    assert "id" in df_sub.columns
    assert "target" in df_sub.columns

    # Check length matches our debug size
    assert len(df_sub) == Config.DEBUG_SAMPLE_SIZE

    print("Training and submission pipeline verified.")


if __name__ == "__main__":
    # Execute the demonstration steps
    demo_data_processing()
    demo_dataset()
    demo_model()
    demo_training_and_submission()

    print("\nAll demonstrations completed successfully.")
