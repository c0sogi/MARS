import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, AverageMeter
from library.data_loader import get_dataloaders, process_data
from library.model import ParallelFactorizedDCNResNet
from library.trainer import run_training


def setup_demo_environment():
    """
    Creates a lightweight environment for demonstration by subsetting the data
    and monkey-patching the Config class to use these subsets and faster settings.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_env"
    os.makedirs(demo_dir, exist_ok=True)

    # Load a tiny fraction of the actual data to preserve schema
    # We read the existing metadata files which are guaranteed to exist
    print("    Subsetting data for speed...")
    try:
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)
        df_test = pd.read_parquet(Config.TEST_PATH)

        # Sample small subsets (e.g., 200 train, 50 val, 50 test)
        # Using a fixed state for reproducibility of the subset
        demo_train = df_train.sample(n=200, random_state=42)
        demo_val = df_val.sample(n=50, random_state=42)
        demo_test = df_test.sample(n=50, random_state=42)

        # Save to demo directory
        demo_train_path = os.path.join(demo_dir, "train.parquet")
        demo_val_path = os.path.join(demo_dir, "val.parquet")
        demo_test_path = os.path.join(demo_dir, "test.parquet")

        demo_train.to_parquet(demo_train_path, index=False)
        demo_val.to_parquet(demo_val_path, index=False)
        demo_test.to_parquet(demo_test_path, index=False)

        print("    Mini-datasets created.")

        # Monkey-patch Config to use these new paths and settings
        Config.TRAIN_PATH = demo_train_path
        Config.VAL_PATH = demo_val_path
        Config.TEST_PATH = demo_test_path

        # Point working directory to demo dir to avoid overwriting real cache
        Config.WORKING_DIR = demo_dir
        Config.CACHE_TRAIN_X = os.path.join(demo_dir, "train_X.npy")
        Config.CACHE_TRAIN_Y = os.path.join(demo_dir, "train_y.npy")
        Config.CACHE_VAL_X = os.path.join(demo_dir, "val_X.npy")
        Config.CACHE_VAL_Y = os.path.join(demo_dir, "val_y.npy")
        Config.CACHE_TEST_X = os.path.join(demo_dir, "test_X.npy")
        Config.CACHE_TEST_IDS = os.path.join(demo_dir, "test_ids.npy")

        # Adjust submission path
        Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

        # Adjust Hyperparameters for speed
        Config.EPOCHS = 1
        Config.BATCH_SIZE = 16
        Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

        # Re-run setup to create directories if needed
        Config.setup()

        print("    Config updated for demo execution.")

    except Exception as e:
        print(f"    Error setting up demo data: {e}")
        sys.exit(1)


def verify_utils():
    """
    Verifies utility functions.
    """
    print("\n>>> Verifying Utils...")

    # Test seed_everything
    seed_everything(123)
    r1 = np.random.rand()
    seed_everything(123)
    r2 = np.random.rand()
    assert r1 == r2, "seed_everything failed to produce reproducible numpy results"
    print("    seed_everything: OK")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    # Total sum = 10*2 + 20*2 = 60. Total count = 4. Avg = 15.
    assert (
        meter.avg == 15.0
    ), f"AverageMeter logic incorrect. Expected 15.0, got {meter.avg}"
    print("    AverageMeter: OK")


def verify_data_loader():
    """
    Verifies data loading and feature engineering.
    """
    print("\n>>> Verifying Data Loader...")

    # Force reload to generate cache from our mini-datasets
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check batch structure
    X_batch, y_batch = next(iter(train_loader))

    # Assertions
    assert isinstance(X_batch, torch.Tensor), "Loader should return X as Tensor"
    assert isinstance(y_batch, torch.Tensor), "Loader should return y as Tensor"
    assert (
        X_batch.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {X_batch.shape[0]}"

    # Feature Engineering Check
    # Original features: 54 (excluding Id, Cover_Type)
    # Engineered features added in `engineer_features`:
    # 1. Aspect_Sin
    # 2. Aspect_Cos
    # 3. Euclidean_Dist_Hydro
    # 4. Abs_Hydro_Elev
    # 5. Mean_Dist_Amenities
    # Total expected features = 54 + 5 = 59.
    expected_dim = 59
    assert (
        X_batch.shape[1] == expected_dim
    ), f"Feature dimension mismatch. Expected {expected_dim}, got {X_batch.shape[1]}"

    print(f"    Data shapes verified. Input Dim: {X_batch.shape[1]}")
    return X_batch.shape[1]


def verify_model(input_dim):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n>>> Verifying Model...")

    num_classes = 7
    model = ParallelFactorizedDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=8,  # Small rank for demo
        hidden_dim=64,
        resnet_blocks=1,
    )
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_dim)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Model output shape mismatch. Expected {(batch_size, num_classes)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("    Model forward pass: OK")


def verify_training_pipeline():
    """
    Verifies the full training pipeline using the trainer module.
    """
    print("\n>>> Verifying Full Training Pipeline...")

    # Run training (Config is already patched to 1 epoch, small batch)
    run_training(epochs=Config.EPOCHS, load_cached_data=True)

    # Verify submission file creation
    sub_path = Config.SUBMISSION_FILE
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Submission file missing required columns"
    assert (
        len(df_sub) == 50
    ), f"Submission length mismatch. Expected 50 (demo test size), got {len(df_sub)}"
    assert (
        df_sub["Cover_Type"].dtype == int or df_sub["Cover_Type"].dtype == np.int64
    ), "Cover_Type must be integer"

    print(f"    Pipeline completed successfully. Submission saved to {sub_path}")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Verify Utils
    verify_utils()

    # 3. Verify Data Loader
    input_dim = verify_data_loader()

    # 4. Verify Model
    verify_model(input_dim)

    # 5. Verify Full Pipeline
    verify_training_pipeline()

    print("\n>>> All verification steps passed successfully.")
