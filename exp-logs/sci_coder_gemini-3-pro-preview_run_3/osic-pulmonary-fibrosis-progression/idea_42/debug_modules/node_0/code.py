import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    laplace_log_likelihood_metric,
    inverse_transform_predictions,
)
from library.data import get_dataloaders
from library.model import SPCRNet
from library.train import MetricAlignedLLLoss, run_training


def demo_metric_logic():
    """
    Demonstrates and validates the custom Laplace Log Likelihood metric.
    """
    print("\n=== Demo 1: Metric Logic Verification ===")

    # Case 1: Perfect prediction
    true_fvc = np.array([2000.0])
    pred_fvc = np.array([2000.0])
    pred_sigma = np.array([100.0])  # Sigma > 70, so not clipped

    # Formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    # Delta = 0
    # Term 1 = 0
    # Term 2 = ln(sqrt(2) * 100) = ln(141.42) approx 4.95
    # Metric = -4.95

    score = laplace_log_likelihood_metric(true_fvc, pred_fvc, pred_sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    print(f"Perfect Prediction Score: {score:.4f}")
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction"

    # Case 2: Clipped Sigma
    # Sigma = 10 (should be clipped to 70)
    pred_sigma_small = np.array([10.0])
    score_clipped = laplace_log_likelihood_metric(true_fvc, pred_fvc, pred_sigma_small)
    expected_score_clipped = -np.log(np.sqrt(2) * 70)

    print(f"Clipped Sigma Score: {score_clipped:.4f}")
    assert np.isclose(
        score_clipped, expected_score_clipped, atol=1e-4
    ), "Metric sigma clipping failed"

    print("Metric logic verified.")


def demo_data_loading():
    """
    Demonstrates data loading and verifies tensor shapes.
    """
    print("\n=== Demo 2: Data Loading & Preprocessing ===")

    # Set Config to Debug to load a small subset
    Config.DEBUG = True
    batch_size = 4

    # Get loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=batch_size, val_batch_size=batch_size
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    images, tabular, targets = batch

    print(f"Batch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Tabular: {tabular.shape}")
    print(f"  Targets: {targets.shape}")

    # Assertions
    # Images: (B, 3, H, W) -> (4, 3, 260, 260)
    assert images.shape == (
        batch_size,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    # Tabular: (B, 5) -> [fvc_norm, age_norm, sex, smoke, time]
    assert tabular.shape == (batch_size, 5), "Incorrect tabular tensor shape"
    # Targets: (B,)
    assert targets.shape == (batch_size,), "Incorrect target tensor shape"

    # Check normalization
    # Tabular features should not be NaN
    assert not torch.isnan(tabular).any(), "Tabular data contains NaNs"
    # Images should be normalized 0..1
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Image normalization out of bounds [0, 1]"

    print("Data loading verified.")
    return images, tabular, targets


def demo_model_forward(images, tabular):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== Demo 3: Model Architecture & Forward Pass ===")

    # Instantiate Model
    model = SPCRNet()
    # Move to CPU for this demo to ensure compatibility if GPU is busy/OOM,
    # though Config.DEVICE handles this usually.
    device = torch.device("cpu")
    model.to(device)

    images = images.to(device)
    tabular = tabular.to(device)

    # Forward Pass
    print("Executing forward pass...")
    final_out, aux_out = model(images, tabular)

    print(f"Output shapes:")
    print(f"  Final Output: {final_out.shape}")
    print(f"  Aux Output:   {aux_out.shape}")

    # Assertions
    # Output should be (B, 2) -> [mu, sigma]
    batch_size = images.size(0)
    assert final_out.shape == (batch_size, 2), "Final output shape mismatch"
    assert aux_out.shape == (batch_size, 2), "Aux output shape mismatch"

    print("Model forward pass verified.")
    return model, final_out, aux_out


def demo_loss_calculation(final_out, targets):
    """
    Demonstrates the custom loss function.
    """
    print("\n=== Demo 4: Loss Calculation ===")

    criterion = MetricAlignedLLLoss()

    # Ensure targets are on the same device
    targets = targets.to(final_out.device)

    # Calculate Loss
    loss = criterion(final_out, targets)

    print(f"Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)"
    assert loss.dim() == 0, "Loss should be a scalar"

    print("Loss calculation verified.")


def demo_full_pipeline():
    """
    Runs the full training pipeline in debug mode for 1 epoch.
    """
    print("\n=== Demo 5: Full Training Pipeline (Debug Mode) ===")

    # Reset Config for the run
    Config.DEBUG = True
    Config.EPOCHS = 1

    # Run training
    # This function handles data loading, model training, validation, and submission generation internally
    run_training(debug=True, epochs=1)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df)} rows.")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in df.columns for col in expected_cols
    ), "Submission missing required columns"

    # Check confidence clipping in submission
    min_conf = df["Confidence"].min()
    print(f"Minimum Confidence in submission: {min_conf}")
    assert min_conf >= 70, "Submission contains confidence values < 70"

    print("Full pipeline verified.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Metric Logic
    demo_metric_logic()

    # 3. Data Loading
    # We keep the batch for the next steps
    images, tabular, targets = demo_data_loading()

    # 4. Model Forward
    model, final_out, aux_out = demo_model_forward(images, tabular)

    # 5. Loss Calculation
    demo_loss_calculation(final_out, targets)

    # 6. Full Pipeline
    # Note: This might take a minute or two depending on CPU/GPU speed
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
