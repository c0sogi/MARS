import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, EarlyStopping
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def create_mini_dataset(source_dir, dest_dir, n_rows=2000):
    """
    Creates a mini dataset by sampling the first n_rows from the original metadata.
    """
    print(f"\n[Demo] Creating mini dataset in {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    files = ["train.parquet", "val.parquet", "test.parquet"]
    for f in files:
        src_path = os.path.join(source_dir, f)
        dest_path = os.path.join(dest_dir, f)

        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            df_mini = df.head(n_rows)
            df_mini.to_parquet(dest_path, index=False)
            print(f"  Created {f}: {df_mini.shape}")
        else:
            raise FileNotFoundError(f"Source file {src_path} not found.")


def test_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n[Demo] Testing Utils...")

    # 1. Test Seed Everything
    seed_everything(42)
    r1 = np.random.rand()
    seed_everything(42)
    r2 = np.random.rand()
    assert r1 == r2, "seed_everything failed to produce reproducible numpy results"
    print("  seed_everything verified.")

    # 2. Test EarlyStopping
    # Create a dummy model to save state
    model = torch.nn.Linear(10, 2)
    early_stopping = EarlyStopping(patience=2, verbose=False)

    # Simulate loss: 0.5 -> 0.4 (improve) -> 0.45 (worse) -> 0.50 (worse, stop)
    losses = [0.5, 0.4, 0.45, 0.50]
    stops = []

    for loss in losses:
        early_stopping(loss, model)
        stops.append(early_stopping.early_stop)

    # Expectation: False, False, False, True (on the 4th step, patience=2 is reached: 0.45(1), 0.50(2))
    assert stops == [
        False,
        False,
        False,
        True,
    ], f"EarlyStopping logic incorrect. Got {stops}"
    assert (
        early_stopping.val_loss_min == 0.4
    ), "EarlyStopping did not record best loss correctly"
    print("  EarlyStopping verified.")


def test_data_pipeline():
    """
    Demonstrates and verifies data loading and processing.
    """
    print("\n[Demo] Testing Data Pipeline...")

    # Force reload to use our mini dataset settings in Config
    # We pass load_cached_data=False to trigger processing logic
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.process_data(
        load_cached_data=False
    )

    # Assertions on Data
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert len(X_train) == 2000, f"Expected 2000 training samples, got {len(X_train)}"
    assert X_train.shape[1] > 50, "Feature engineering should result in >50 features"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"

    # Check Feature Engineering specific columns (based on logic in data_loader.py)
    # Original cols ~54. Added: Aspect_Sin, Aspect_Cos, Euclidean_Dist, Abs_Hydro, Mean_Dist (5 cols)
    # One-hot encoding isn't done, but binary cols are preserved.
    # Standard scaling is applied to continuous.

    print(f"  Data Processed. X_train shape: {X_train.shape}")

    # Test DataLoaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Now we can load the cache we just made
        num_workers=0,  # Use 0 workers for simple debugging/demo
    )

    batch_X, batch_y = next(iter(train_loader))
    assert batch_X.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_X.shape[1] == X_train.shape[1], "Feature dimension mismatch in loader"
    assert batch_y.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("  DataLoaders verified.")
    return X_train.shape[1]


def test_model_architecture(input_dim):
    """
    Demonstrates and verifies the model architecture.
    """
    print("\n[Demo] Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = model_lib.DeepParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=64,  # Small hidden dim for test
        num_blocks=2,
        dropout=0.1,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Create dummy input
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check output
    assert output.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {output.shape}"

    print("  Model forward pass verified.")


def test_training_loop():
    """
    Demonstrates the full training loop using the library function.
    """
    print("\n[Demo] Testing Training Loop...")

    # We use the parameters defined in Config (which we modified in main)
    # run_training handles everything: loading data, init model, training, saving best
    model = train_lib.run_training(
        load_cached_data=True,  # Use the cache from previous step
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        hidden_dim=128,
        num_blocks=1,
    )

    # Check if model file was created
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved"

    print("  Training loop completed and model saved.")


def main():
    # 1. Setup Directories and Config
    DEMO_ROOT = "./working/demo_execution"
    DEMO_DATA_DIR = os.path.join(DEMO_ROOT, "data")
    DEMO_CACHE_DIR = os.path.join(DEMO_ROOT, "cache")

    # Clean start
    if os.path.exists(DEMO_ROOT):
        shutil.rmtree(DEMO_ROOT)
    os.makedirs(DEMO_DATA_DIR)

    # 2. Create Mini Dataset for Speed
    create_mini_dataset("./metadata", DEMO_DATA_DIR, n_rows=2000)

    # 3. Override Config for Demonstration
    # We modify the class attributes directly to affect all modules importing it
    Config.INPUT_DIR = "./input"  # Remains same
    Config.METADATA_DIR = DEMO_DATA_DIR  # Point to mini metadata
    Config.WORKING_DIR = DEMO_CACHE_DIR

    # Point paths to the mini parquet files
    Config.TRAIN_PATH = os.path.join(DEMO_DATA_DIR, "train.parquet")
    Config.VAL_PATH = os.path.join(DEMO_DATA_DIR, "val.parquet")
    Config.TEST_PATH = os.path.join(DEMO_DATA_DIR, "test.parquet")

    # Reduce compute requirements for demo
    Config.BATCH_SIZE = 64
    Config.EPOCHS = 2
    Config.PATIENCE = 2
    Config.NUM_CLASSES = 7  # As per dataset (Cover_Type 1-7)

    # 4. Run Tests
    try:
        test_utils()
        input_dim = test_data_pipeline()
        test_model_architecture(input_dim)
        test_training_loop()

        print("\n[Demo] All demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\n[Demo] FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Demo] ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
