import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import NDSSLN
from library.train import run_training


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("1. Setting up configuration...")

    # Set random seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demonstration
    # We use class attribute modification to affect the library globally
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Define temporary working directories for this demo
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.CHECKPOINT_DIR = "./working/demo_execution/checkpoints"
    Config.SUBMISSION_PATH = "./working/demo_execution/submission.csv"

    # Ensure these directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n2. Verifying Data Pipeline...")

    # Initialize DataLoaders with debug=True to use a subset of data
    # This loads the first 50 training samples and 20 validation samples
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    print(f"   Train Batches: {len(train_loader)}")
    print(f"   Val Batches:   {len(val_loader)}")

    # Fetch a single batch to verify structure and shapes
    try:
        batch = next(iter(train_loader))
        img_ax = batch["img_ax"]
        img_cor = batch["img_cor"]
        tabular = batch["tabular"]
        target = batch["target"]

        # Verify Shapes
        # Images: (B, 3, 224, 224)
        expected_img_shape = (Config.BATCH_SIZE, 3, 224, 224)
        assert (
            img_ax.shape == expected_img_shape
        ), f"Axial Image shape mismatch: {img_ax.shape}"
        assert (
            img_cor.shape == expected_img_shape
        ), f"Coronal Image shape mismatch: {img_cor.shape}"

        # Tabular: (B, 6) -> [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent]
        assert tabular.shape == (
            Config.BATCH_SIZE,
            6,
        ), f"Tabular shape mismatch: {tabular.shape}"

        # Target: (B, 1)
        assert target.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Target shape mismatch: {target.shape}"

        print("   Data shapes verified successfully.")

    except Exception as e:
        print(f"   Data Pipeline Failed: {e}")
        raise e

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n3. Verifying Model Architecture...")

    try:
        device = Config.DEVICE
        model = NDSSLN().to(device)

        # Move batch data to the appropriate device
        img_ax = img_ax.to(device)
        img_cor = img_cor.to(device)
        tabular = tabular.to(device)

        # Perform a forward pass
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Verify Output Shapes
        # All outputs should be (B, 1)
        assert alpha.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Alpha shape mismatch: {alpha.shape}"
        assert sigma_base.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Sigma Base shape mismatch: {sigma_base.shape}"
        assert sigma_growth.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Sigma Growth shape mismatch: {sigma_growth.shape}"

        # Verify Positivity Constraints (Sigma must be positive due to Softplus)
        assert (sigma_base > 0).all(), "Sigma Base contains non-positive values"
        assert (sigma_growth > 0).all(), "Sigma Growth contains non-positive values"

        print("   Model forward pass verified successfully.")

    except Exception as e:
        print(f"   Model Verification Failed: {e}")
        raise e

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n4. Executing Training Loop (Debug Mode)...")

    # run_training handles the full lifecycle:
    # 1. Training for N epochs
    # 2. Validation and Checkpointing
    # 3. Loading best model
    # 4. Generating submission
    run_training(epochs=Config.EPOCHS, debug=True)

    # ==========================================
    # 5. Output Verification
    # ==========================================
    print("\n5. Verifying Outputs...")

    # Verify Checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"   Checkpoint found: {best_model_path}")
        # Verify file size is non-zero
        assert os.path.getsize(best_model_path) > 0, "Checkpoint file is empty"
    else:
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}")

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"   Submission found: {Config.SUBMISSION_PATH}")
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)

        # Check Dimensions
        print(f"   Submission Rows: {len(sub_df)}")

        # Check Columns
        required_cols = ["Patient_Week", "FVC", "Confidence"]
        missing_cols = [c for c in required_cols if c not in sub_df.columns]
        assert not missing_cols, f"Submission missing columns: {missing_cols}"

        # Check for NaNs
        assert not sub_df.isnull().values.any(), "Submission contains NaN values"

        # Check Data Types
        assert pd.api.types.is_numeric_dtype(sub_df["FVC"]), "FVC column is not numeric"
        assert pd.api.types.is_numeric_dtype(
            sub_df["Confidence"]
        ), "Confidence column is not numeric"

        print("   Submission format verified successfully.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
