import os
import sys
import numpy as np
import pandas as pd
import torch

# Import classes and functions from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data
from library.model import IcebergResNet18
from library.train import train_bagging_ensemble
from library.inference import generate_submission


def run_demo():
    print("=== Iceberg Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set a specific directory for this demo run to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce hyperparameters to ensure quick execution
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch per bag
    Config.NUM_BAGS = 2  # Use only 2 bags for the ensemble
    Config.BATCH_SIZE = 16  # Reduce batch size

    # Create necessary directories
    Config.make_dirs()

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Submission Path: {Config.SUBMISSION_PATH}")
    print(f"Configuration: {Config.NUM_BAGS} Bags, {Config.NUM_EPOCHS} Epochs")

    # ---------------------------------------------------------
    # 2. Data Loading and Verification
    # ---------------------------------------------------------
    print("\n[2] Loading and processing data...")

    # load_data handles caching. Since we changed WORKING_DIR, it will process from scratch.
    train_images, train_angles, train_labels, test_images, test_angles, test_ids = (
        load_data(load_cached_data=True)
    )

    # Verify Data Shapes and Integrity
    print(f"Train Images: {train_images.shape}")
    print(f"Train Angles: {train_angles.shape}")
    print(f"Train Labels: {train_labels.shape}")

    # Assertions
    assert (
        len(train_images) == len(train_angles) == len(train_labels)
    ), "Mismatch in training data lengths"
    assert train_images.shape[1:] == (
        224,
        224,
        3,
    ), f"Unexpected image shape: {train_images.shape}"
    assert not np.isnan(train_images).any(), "Training images contain NaNs"
    assert not np.isnan(train_angles).any(), "Training angles contain NaNs"

    print("Data verification passed.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying model architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = IcebergResNet18().to(device)

    # Create dummy inputs matching the expected batch format
    dummy_images = torch.randn(4, 3, 224, 224).to(device)
    dummy_angles = torch.randn(4, 1).to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_images, dummy_angles)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (4, 1), f"Expected output shape (4, 1), got {logits.shape}"
    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("\n[4] Starting Ensemble Training...")

    # This function uses the Config parameters we modified (NUM_BAGS, NUM_EPOCHS)
    train_bagging_ensemble()

    # Verify that checkpoints were created
    for i in range(Config.NUM_BAGS):
        ckpt_path = os.path.join(Config.WORKING_DIR, f"model_bag_{i}.pth")
        if os.path.exists(ckpt_path):
            print(f"Checkpoint found: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint for bag {i} was not created!")

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    print("\n[5] Generating Submission...")

    # Generates submission.csv using the trained models
    generate_submission()

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    print("Inference complete.")

    # ---------------------------------------------------------
    # 6. Submission Validation
    # ---------------------------------------------------------
    print("\n[6] Validating Submission File...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission Dimensions: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Assertions
    assert list(df_sub.columns) == ["id", "is_iceberg"], "Incorrect column names"
    assert len(df_sub) == len(
        test_ids
    ), f"Expected {len(test_ids)} rows, got {len(df_sub)}"
    assert df_sub["is_iceberg"].min() >= 0.0, "Probabilities < 0 found"
    assert df_sub["is_iceberg"].max() <= 1.0, "Probabilities > 1 found"
    assert df_sub["id"].nunique() == len(df_sub), "Duplicate IDs found"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
