import os
import sys
import warnings
import torch
import numpy as np
import pandas as pd

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config, set_seed
from library.dataset import SETIDataset
from library.model import ShallowCNN
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Technosignature Detection Library Demo ===")

    # 1. Setup and Configuration Override for Speed
    # We override specific Config attributes to ensure the demo runs quickly
    # and fits within the compute/time constraints.
    print("\n[Step 1] Configuring environment for fast demonstration...")
    set_seed(42)

    # Enable debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small enough for instant processing

    # Reduce training duration
    Config.EPOCHS = 1
    Config.PATIENCE = 1

    # Ensure working directories exist (Config.setup() does this, but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(
        f"Configuration: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, DEVICE={Config.DEVICE}"
    )

    # 2. Dataset Verification
    print("\n[Step 2] Verifying SETIDataset...")

    # Initialize dataset
    train_dataset = SETIDataset(metadata_path=Config.TRAIN_METADATA)

    # Fetch one sample
    sample_idx = 0
    image_tensor, target_tensor = train_dataset[sample_idx]

    # Assertions
    print(f"Sample {sample_idx} shape: {image_tensor.shape}")
    print(f"Sample {sample_idx} target: {target_tensor}")

    # Check dimensions: (6, 273, 256)
    assert image_tensor.shape == (
        6,
        273,
        256,
    ), f"Expected shape (6, 273, 256), got {image_tensor.shape}"

    # Check target type
    assert isinstance(target_tensor, torch.Tensor), "Target should be a torch.Tensor"

    # Check Normalization (Mean ~ 0, Std ~ 1)
    # Note: Instance normalization is per sample.
    mean_val = image_tensor.mean().item()
    std_val = image_tensor.std().item()
    print(f"Sample Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")

    assert (
        abs(mean_val) < 1e-4
    ), f"Normalization failed: Mean is {mean_val}, expected ~0"
    assert (
        abs(std_val - 1.0) < 1e-4
    ), f"Normalization failed: Std is {std_val}, expected ~1"

    print("Dataset verification passed.")

    # 3. Model Verification
    print("\n[Step 3] Verifying ShallowCNN Model...")

    model = ShallowCNN()
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy batch of size 2
    dummy_input = image_tensor.unsqueeze(0).repeat(2, 1, 1, 1).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    # Assert output shape (Batch_Size, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model verification passed.")

    # 4. Training Loop Demonstration
    print("\n[Step 4] Demonstrating Training Loop (Debug Mode)...")

    trainer = Trainer()

    # Run training
    # This uses the logic in trainer.fit(), which respects Config.DEBUG
    trainer.fit()

    # Verify that the model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"

    print("Training demonstration passed. Model saved.")

    # 5. Inference Demonstration
    print("\n[Step 5] Demonstrating Inference and Submission Generation...")

    # Generate submission using the model we just trained
    # Note: This runs on the full test set (6000 samples).
    # With a shallow CNN and large batch size, this is fast enough.
    generate_submission(
        model_path=best_model_path,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE * 2,  # Increase batch size for faster inference
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(f"Columns: {list(df_sub.columns)}")

    # Check columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission file missing required columns."

    # Check row count matches test metadata
    df_test_meta = pd.read_csv(Config.TEST_METADATA)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) does not match test set size ({len(df_test_meta)})."

    # Check value range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Predictions contain values outside [0, 1]."

    print("Inference demonstration passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
