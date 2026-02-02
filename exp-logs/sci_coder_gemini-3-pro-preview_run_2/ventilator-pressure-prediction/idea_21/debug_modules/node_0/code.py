import os
import shutil
import pandas as pd
import numpy as np
import torch
import sys

# Import from the provided library
from library.config import Config
from library.utils import set_seed, WeightedL1Loss
from library.dataset import Preprocessor, VentilatorDataset, load_data
from library.model import FPBC_BiLSTM
from library.train import Trainer


def setup_demo_environment():
    """
    Sets up a demo environment by creating a mini-dataset and patching the Config.
    """
    print(">>> Setting up demo environment...")

    # 1. Define Demo Paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Create Mini Datasets (Subset of real data)
    # Read a small chunk of train.csv
    # We need enough rows to get full breaths (80 steps per breath)
    # 20 breaths * 80 steps = 1600 rows
    df_train_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"), nrows=3000)

    # Get first 20 unique breath IDs
    breath_ids = df_train_full["breath_id"].unique()[:20]
    df_mini_train = df_train_full[df_train_full["breath_id"].isin(breath_ids)].copy()

    # Split: 16 Train, 4 Val
    train_ids = set(breath_ids[:16])
    val_ids = set(breath_ids[16:])

    # Save Mini Train CSV
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    df_mini_train.to_csv(mini_train_path, index=False)

    # Create Metadata Files
    # Train Meta
    df_mini_train_meta = df_mini_train[
        df_mini_train["breath_id"].isin(train_ids)
    ].copy()
    df_mini_train_meta["source_file"] = (
        "mini_train.csv"  # Not strictly used by load_data logic but good for consistency
    )
    mini_train_meta_path = os.path.join(demo_dir, "mini_train_meta.csv")
    df_mini_train_meta.to_csv(mini_train_meta_path, index=False)

    # Val Meta
    df_mini_val_meta = df_mini_train[df_mini_train["breath_id"].isin(val_ids)].copy()
    df_mini_val_meta["source_file"] = "mini_train.csv"
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_meta.csv")
    df_mini_val_meta.to_csv(mini_val_meta_path, index=False)

    # Test Data (5 breaths)
    df_test_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"), nrows=1000)
    test_breath_ids = df_test_full["breath_id"].unique()[:5]
    df_mini_test = df_test_full[df_test_full["breath_id"].isin(test_breath_ids)].copy()

    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    df_mini_test.to_csv(mini_test_path, index=False)

    # 3. Patch Config
    # We modify the Config class attributes directly to redirect the pipeline
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_DATA_PATH = mini_train_path
    Config.TEST_DATA_PATH = mini_test_path
    Config.TRAIN_META_PATH = mini_train_meta_path
    Config.VAL_META_PATH = mini_val_meta_path

    # Update dependent paths (since they were initialized at import time)
    Config.TRAIN_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.SCALER_CACHE_PATH = os.path.join(Config.WORKING_DIR, "scaler_params.npy")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Optimize Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for tiny data
    Config.LSTM_HIDDEN_SIZE = 32  # Reduce model size
    Config.LSTM_LAYERS = 2
    Config.GLU_WIDE_SIZE = 32
    Config.CONTEXT_BOTTLENECK_SIZE = 16

    # Ensure directories exist
    Config.setup()
    print(">>> Demo environment ready.")


def verify_loss_function():
    """
    Verifies the WeightedL1Loss logic.
    """
    print(">>> Verifying WeightedL1Loss...")
    criterion = WeightedL1Loss()

    # Create dummy data: 2 samples
    # Sample 1: Inspiratory (u_out=0), Error=10
    # Sample 2: Expiratory (u_out=1), Error=10
    preds = torch.tensor([10.0, 10.0])
    targets = torch.tensor([20.0, 20.0])
    u_out = torch.tensor([0.0, 1.0])

    # Expected Loss:
    # Insp Weight = 1.0, Exp Weight = 0.1 (Default in Config)
    # L1 Raw = |10-20| = 10
    # Loss 1 = 10 * 1.0 = 10
    # Loss 2 = 10 * 0.1 = 1
    # Mean = (10 + 1) / 2 = 5.5

    loss = criterion(preds, targets, u_out)

    assert (
        abs(loss.item() - 5.5) < 1e-6
    ), f"Loss calculation incorrect. Expected 5.5, got {loss.item()}"
    print(">>> WeightedL1Loss verified.")


def verify_model_architecture():
    """
    Verifies the FPBC_BiLSTM model output shape.
    """
    print(">>> Verifying FPBC_BiLSTM architecture...")
    model = FPBC_BiLSTM()
    model.eval()

    # Batch size 2, Seq len 80, Input dim (from Config)
    # Config.INPUT_DIM is calculated based on features.
    # Continuous (9) + Binary (1) = 10 features.
    input_dim = Config.INPUT_DIM

    x = torch.randn(2, 80, input_dim)
    u_out = torch.zeros(2, 80)  # Not used by model logic but passed in forward

    with torch.no_grad():
        out = model(x, u_out)

    # Expected output: (Batch, Seq) -> (2, 80)
    assert out.shape == (
        2,
        80,
    ), f"Model output shape mismatch. Expected (2, 80), got {out.shape}"
    print(">>> Model architecture verified.")


def run_pipeline():
    """
    Runs the Trainer to demonstrate the full workflow.
    """
    print(">>> Starting Training Pipeline...")

    # Initialize Trainer
    trainer = Trainer()

    # Run Fit
    # This handles:
    # 1. load_data (which triggers Preprocessor, Feature Engineering, Caching)
    # 2. Training Loop (1 Epoch)
    # 3. Validation
    # 4. Submission Generation
    trainer.fit()

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # We used 5 breaths for test, 80 steps each = 400 rows
    expected_rows = 5 * 80
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print(
        f">>> Pipeline completed successfully. Submission generated at {Config.SUBMISSION_PATH}"
    )
    print(df_sub.head())


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Verify Components
    verify_loss_function()
    verify_model_architecture()

    # 3. Run Pipeline
    run_pipeline()
