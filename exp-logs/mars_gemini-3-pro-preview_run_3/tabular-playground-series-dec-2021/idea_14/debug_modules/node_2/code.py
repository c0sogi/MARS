import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model_utils import ParallelLowRankDCNResNet, predict_and_submit
from library.train_utils import run_training


def create_mini_datasets():
    """
    Creates small subsets of the original data to ensure the demo runs quickly.
    Modifies Config paths to point to these new subsets.
    """
    print("Creating mini datasets for rapid demonstration...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.parquet")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.parquet")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.parquet")

    # Load head of original parquet files
    # We use a small number of rows (e.g., 2000) to make feature engineering instant
    df_train = pd.read_parquet(Config.TRAIN_PATH).head(2000)
    df_val = pd.read_parquet(Config.VAL_PATH).head(1000)
    df_test = pd.read_parquet(Config.TEST_PATH).head(1000)

    # Save mini datasets
    df_train.to_parquet(mini_train_path, index=False)
    df_val.to_parquet(mini_val_path, index=False)
    df_test.to_parquet(mini_test_path, index=False)

    print(f"Mini datasets saved to {Config.WORKING_DIR}")

    # Update Config to use these files
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path

    # Update Cache Directory to avoid conflicts with real run or loading large files
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)


def configure_demo_settings():
    """
    Overrides default Config hyperparameters for speed.
    """
    print("Overriding Config hyperparameters...")
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 32  # Small batch size
    Config.RESNET_WIDTH = 64  # Smaller model width
    Config.RESNET_LAYERS = 1  # Shallower network
    Config.DCN_RANK = 4  # Lower rank
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.DEBUG = True  # Enable debug mode logic if applicable
    Config.DEBUG_SAMPLE_SIZE = 2000  # Align with our mini dataset size


def main():
    # 1. Setup Environment
    Config.set_seed(42)
    create_mini_datasets()
    configure_demo_settings()

    # 2. Test Data Loading
    print("\n=== Testing Data Pipeline ===")
    # load_cached_data=False ensures we process our new mini datasets
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False,
        debug=False,  # We already manually subsetted the files, so debug=False avoids double slicing
    )

    # Verify DataLoaders
    try:
        batch_X, batch_y = next(iter(train_loader))
        print(f"Train Batch Shape: X={batch_X.shape}, y={batch_y.shape}")

        # Assertions
        assert batch_X.dim() == 2, "Input batch must be 2D (Batch, Features)"
        assert batch_y.dim() == 1, "Target batch must be 1D (Batch)"
        assert batch_X.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

        input_dim = batch_X.shape[1]
        print(f"Feature Dimension detected: {input_dim}")

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Test Model Architecture
    print("\n=== Testing Model Architecture ===")
    model = ParallelLowRankDCNResNet(
        input_dim=input_dim, num_classes=Config.NUM_CLASSES
    )
    model.to(Config.DEVICE)

    # Test Forward Pass
    with torch.no_grad():
        dummy_input = batch_X.to(Config.DEVICE)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN outputs"

    # 4. Test Training Loop
    print("\n=== Testing Training Loop ===")
    # run_training handles optimizer, scheduler, and early stopping
    trained_model = run_training(model, train_loader, val_loader)

    assert isinstance(
        trained_model, torch.nn.Module
    ), "run_training did not return a model"
    print("Training loop executed successfully.")

    # 5. Test Prediction and Submission
    print("\n=== Testing Prediction and Submission ===")
    predict_and_submit(trained_model, test_loader, test_ids)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Assertions on Submission
    assert list(df_sub.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Incorrect submission columns"
    assert len(df_sub) == len(
        test_ids
    ), f"Submission length mismatch. Expected {len(test_ids)}, got {len(df_sub)}"
    assert df_sub[Config.TARGET_COL].dtype in [
        np.int64,
        int,
    ], "Target column should be integers"

    # Verify values are within expected range (1-7)
    # Note: Model outputs 0-6 internally, but predict_and_submit adds 1.
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 1 and preds.max() <= 7
    ), "Predictions out of valid class range (1-7)"

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
