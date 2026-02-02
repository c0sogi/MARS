import os
import shutil
import warnings
import pandas as pd
import torch
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import DeepPreActDCNResNet
from library.train import Trainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("Initializing demo pipeline...")
    warnings.filterwarnings("ignore")
    seed_everything(42)

    # Define a temporary working directory for this demo
    demo_dir = "./working/demo_pipeline"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # Create necessary subdirectories
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "submission"), exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Create Mini Datasets (Optimization for Speed)
    # -------------------------------------------------------------------------
    print("Creating mini-datasets for rapid verification...")

    # Load a small subset of the actual metadata
    # We use the paths defined in the default Config to find the source files
    train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_full = pd.read_parquet(Config.VAL_DATA_PATH)
    test_full = pd.read_parquet(Config.TEST_DATA_PATH)

    # Create subsets: 2000 train, 500 val, 500 test
    mini_train = train_full.head(2000)
    mini_val = val_full.head(500)
    mini_test = test_full.head(500)

    # Save these mini-datasets to the demo directory
    mini_train_path = os.path.join(demo_dir, "mini_train.parquet")
    mini_val_path = os.path.join(demo_dir, "mini_val.parquet")
    mini_test_path = os.path.join(demo_dir, "mini_test.parquet")

    mini_train.to_parquet(mini_train_path, index=False)
    mini_val.to_parquet(mini_val_path, index=False)
    mini_test.to_parquet(mini_test_path, index=False)

    # -------------------------------------------------------------------------
    # 3. Override Configuration
    # -------------------------------------------------------------------------
    print("Overriding configuration for demo execution...")

    # Update paths to point to our mini datasets and demo directories
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    Config.TRAIN_DATA_PATH = mini_train_path
    Config.VAL_DATA_PATH = mini_val_path
    Config.TEST_DATA_PATH = mini_test_path

    # Update Cache Paths to avoid overwriting or reading production cache
    Config.CACHE_TRAIN_X = os.path.join(demo_dir, "cache", "X_train.npy")
    Config.CACHE_TRAIN_Y = os.path.join(demo_dir, "cache", "y_train.npy")
    Config.CACHE_VAL_X = os.path.join(demo_dir, "cache", "X_val.npy")
    Config.CACHE_VAL_Y = os.path.join(demo_dir, "cache", "y_val.npy")
    Config.CACHE_TEST_X = os.path.join(demo_dir, "cache", "X_test.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "cache", "test_ids.npy")

    # Update Output Paths
    Config.MODEL_CHECKPOINT_PATH = os.path.join(demo_dir, "cache", "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 256
    Config.HIDDEN_DIM = 64  # Smaller model
    Config.NUM_RESNET_BLOCKS = 2  # Shallower network
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # -------------------------------------------------------------------------
    # 4. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("Executing Data Pipeline...")

    # load_cached_data=False forces the pipeline to process the raw parquet files
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verify Data Dimensions
    print(f"   Input Dimension (dynamically set): {Config.INPUT_DIM}")
    assert (
        len(train_loader.dataset) == 2000
    ), f"Train set size mismatch. Expected 2000, got {len(train_loader.dataset)}"
    assert (
        len(val_loader.dataset) == 500
    ), f"Val set size mismatch. Expected 500, got {len(val_loader.dataset)}"
    assert (
        len(test_loader.dataset) == 500
    ), f"Test set size mismatch. Expected 500, got {len(test_loader.dataset)}"
    assert hasattr(
        Config, "INPUT_DIM"
    ), "Config.INPUT_DIM was not set by get_dataloaders"

    # -------------------------------------------------------------------------
    # 5. Model Verification
    # -------------------------------------------------------------------------
    print("Initializing Model...")

    device = get_device()
    model = DeepPreActDCNResNet()
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(10, Config.INPUT_DIM).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        10,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (10, {Config.NUM_CLASSES}), got {dummy_output.shape}"
    print("   Forward pass check passed.")

    # -------------------------------------------------------------------------
    # 6. Training Verification
    # -------------------------------------------------------------------------
    print("Starting Training Loop...")

    trainer = Trainer(model, device=device)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify Checkpoint Creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."
    print("   Training complete and checkpoint verified.")

    # -------------------------------------------------------------------------
    # 7. Inference Verification
    # -------------------------------------------------------------------------
    print("Generating Predictions...")

    trainer.predict(test_loader, test_ids)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert submission_df.shape == (
        500,
        2,
    ), f"Submission shape mismatch. Expected (500, 2), got {submission_df.shape}"
    assert list(submission_df.columns) == [
        "Id",
        "Cover_Type",
    ], "Submission columns mismatch."

    # Check if predictions are valid classes (1-7)
    preds = submission_df["Cover_Type"].unique()
    assert all(
        p in range(1, 8) for p in preds
    ), f"Invalid class predictions found: {preds}"

    print(f"   Submission saved to {Config.SUBMISSION_PATH}")
    print("\nAll pipeline components verified successfully.")


if __name__ == "__main__":
    main()
