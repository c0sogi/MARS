import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import SBPDSNet
from library.train import train_model, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.CACHE_DIR = "./working/demo_task_execution/cache/"
    Config.SUBMISSION_PATH = "./working/demo_task_execution/submission/submission.csv"
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.EPOCHS = 2

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration updated. Random seeds set.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Load a tiny subset of data
    # We use 16 training samples and 8 validation samples for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        max_train_samples=16,
        max_val_samples=8,
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches:   {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    targets = batch["target"]
    pids = batch["patient_week"]

    print(
        f"Batch Shapes -> Image: {images.shape}, Tabular: {tabular.shape}, Target: {targets.shape}"
    )

    # Assertions to verify data integrity
    # Image: (B, 3, 260, 260)
    assert images.dim() == 4, "Image tensor should be 4D (B, C, H, W)"
    assert images.shape[1] == 3, "Image tensor should have 3 channels (slices)"
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Image size mismatch. Expected {Config.IMG_SIZE}x{Config.IMG_SIZE}"

    # Tabular: (B, 5) -> [Baseline_FVC, Time, Age, Sex, Smoking]
    assert tabular.dim() == 2, "Tabular tensor should be 2D (B, Features)"
    assert tabular.shape[1] == 5, "Tabular features should have 5 dimensions"

    # Target: (B,)
    assert targets.dim() == 1, "Target tensor should be 1D"

    print("Data loading verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SBPDSNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    outputs = model(images, tabular)

    # Verify outputs
    required_keys = ["final_mean", "final_sigma", "base_mean", "base_sigma"]
    for key in required_keys:
        assert key in outputs, f"Model output missing key: {key}"
        assert (
            outputs[key].shape[0] == Config.BATCH_SIZE
        ), f"Output batch size mismatch for {key}"

    # Check that sigma is positive (Softplus used in model)
    assert (outputs["final_sigma"] > 0).all(), "Final Sigma must be positive"
    assert (outputs["base_sigma"] > 0).all(), "Base Sigma must be positive"

    print(f"Model forward pass successful on {device}.")

    # -------------------------------------------------------------------------
    # 4. Metric Calculation Check
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Metric Calculation...")

    # Create synthetic data
    # Case 1: Perfect prediction
    y_true = np.array([2000.0, 3000.0])
    y_pred = np.array([2000.0, 3000.0])
    sigma = np.array([100.0, 100.0])  # > 70, so not clipped

    # Formula: - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100)
    #        = 0 - ln(141.42...)
    #        = -4.95...
    score_perfect = calculate_metric(y_true, y_pred, sigma)

    # Case 2: Large error (clipped at 1000)
    y_pred_bad = np.array([4000.0, 5000.0])  # Error 2000 -> clipped to 1000
    # Formula: - (sqrt(2) * 1000) / 100 - ln(sqrt(2) * 100)
    #        = - 14.142... - 4.95...
    #        = -19.09...
    score_bad = calculate_metric(y_true, y_pred_bad, sigma)

    print(f"Score (Perfect): {score_perfect:.4f}")
    print(f"Score (Bad):     {score_bad:.4f}")

    assert (
        score_perfect > score_bad
    ), "Metric logic error: Perfect score should be higher than bad score."
    print("Metric calculation verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Short Run)...")

    # Train for 2 epochs on the small subset
    # Note: train_model internally calls get_dataloaders, so we pass the sample limits there
    best_model_path, scalers = train_model(
        epochs=Config.EPOCHS, max_train_samples=32, max_val_samples=16
    )

    print(f"Training finished. Best model saved at: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file was not created."
    assert "fvc_mean" in scalers, "Scalers dictionary missing 'fvc_mean'"

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    generate_submission(best_model_path, scalers)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print("First 3 rows:")
    print(sub_df.head(3))

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check if we have predictions for the test set
    # sample_submission has 1909 rows (header included in count logic usually, pandas shape is rows)
    # The provided sample_submission.csv has 1908 data rows.
    # Our generate_submission logic expands the test set based on sample_submission.csv.
    # So the output length should match sample_submission.csv length.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    assert len(sub_df) == len(
        sample_sub
    ), f"Submission length mismatch. Expected {len(sample_sub)}, got {len(sub_df)}"

    print("\n=== Demonstration Complete. All checks passed. ===")


if __name__ == "__main__":
    main()
