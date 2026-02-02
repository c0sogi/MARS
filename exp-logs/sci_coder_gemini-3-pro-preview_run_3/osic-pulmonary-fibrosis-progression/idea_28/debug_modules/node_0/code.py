import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import MACAN
from library.train import train_model
from library.inference import predict_test


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Override for Speed and Demo Purposes
    # We modify the Config class attributes directly to run a fast, minimal version.
    print("Configuring for demo execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 patients for training/val
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small script
    Config.WORKING_DIR = "./working/demo_execution"

    # Update derived paths based on the new working directory
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Set random seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Pipeline Verification
    print("\n=== Verifying Data Pipeline ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch from training loader
    batch = next(iter(train_loader))

    images = batch["image"]
    tabular = batch["tabular"]
    targets = batch["target"]

    print(f"Batch Image Shape: {images.shape}")  # Expected: (B, 3, 260, 260)
    print(f"Batch Tabular Shape: {tabular.shape}")  # Expected: (B, 8)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (B, 1)

    # Assertions for Data Shapes
    assert images.ndim == 4, "Image tensor should be 4D (B, C, H, W)"
    assert images.shape[1] == 3, "Image tensor should have 3 channels (slices)"
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Image size mismatch. Expected {Config.IMG_SIZE}x{Config.IMG_SIZE}"
    assert tabular.shape[1] == 8, "Tabular data should have 8 features"
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("Data pipeline verification passed.")

    # 3. Model Architecture Verification
    print("\n=== Verifying Model Architecture ===")
    model = MACAN().to(device)

    # Move batch to device
    images_dev = images.to(device)
    tabular_dev = tabular.to(device)

    # Forward pass
    mu, sigma = model(images_dev, tabular_dev)

    print(f"Output Mu Shape: {mu.shape}")  # Expected: (B,)
    print(f"Output Sigma Shape: {sigma.shape}")  # Expected: (B,)

    # Assertions for Model Output
    assert mu.shape[0] == Config.BATCH_SIZE, "Output batch size mismatch for Mu"
    assert sigma.shape[0] == Config.BATCH_SIZE, "Output batch size mismatch for Sigma"

    # Check positivity of Sigma (since Softplus is used)
    if not torch.all(sigma > 0):
        raise AssertionError("Sigma predictions must be positive.")

    print("Model architecture verification passed.")

    # 4. Metric Calculation Verification
    print("\n=== Verifying Metric Calculation ===")
    # Create dummy data
    # Case: Perfect prediction
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma_pred = np.array([100.0])  # > 70, so not clipped

    # Expected metric:
    # Delta = 0
    # Term 1 = 0
    # Term 2 = ln(sqrt(2) * 100) = ln(141.42...) ≈ 4.95
    # Result = -4.95
    score = calculate_metric(y_true, y_pred, sigma_pred)
    print(f"Metric Score (Perfect Prediction, Sigma=100): {score:.4f}")

    # Case: Error with clipping
    y_true_bad = np.array([2000.0])
    y_pred_bad = np.array([3500.0])  # Diff 1500 -> Clipped to 1000
    sigma_bad = np.array([50.0])  # < 70 -> Clipped to 70

    # Expected:
    # Delta = 1000
    # Sigma_clipped = 70
    # Term 1 = (sqrt(2) * 1000) / 70 ≈ 1414.21 / 70 ≈ 20.20
    # Term 2 = ln(sqrt(2) * 70) ≈ ln(98.99) ≈ 4.60
    # Result = -(20.20 + 4.60) = -24.80
    score_bad = calculate_metric(y_true_bad, y_pred_bad, sigma_bad)
    print(f"Metric Score (Large Error, Low Sigma): {score_bad:.4f}")

    assert (
        score > score_bad
    ), "Better prediction should have higher (less negative) score"
    print("Metric verification passed.")

    # 5. Training Loop Demonstration
    print("\n=== Running Training Loop (Demo) ===")
    # This will run for 1 epoch on the small debug subset
    train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # Verify checkpoint creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )
    print(f"Training complete. Checkpoint saved to {Config.BEST_MODEL_PATH}")

    # 6. Inference Demonstration
    print("\n=== Running Inference (Demo) ===")
    # This uses the trained model to generate predictions for the test set
    predict_test()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {submission_df.shape}")
    print("First 5 rows:")
    print(submission_df.head())

    # Assertions on Submission
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        if col not in submission_df.columns:
            raise AssertionError(f"Submission missing required column: {col}")

    # Check for NaNs
    if submission_df.isnull().any().any():
        raise AssertionError("Submission contains NaN values.")

    # Check if FVC and Confidence are numeric
    if not pd.api.types.is_numeric_dtype(submission_df["FVC"]):
        raise AssertionError("FVC column must be numeric.")

    print("Inference verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
