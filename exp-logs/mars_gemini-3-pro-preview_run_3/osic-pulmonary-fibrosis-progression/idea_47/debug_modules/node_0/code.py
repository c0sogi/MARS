import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, metric_score, InverseScaler, AverageMeter
from library.loss import LaplaceNLLLoss
from library.data import get_dataloaders
from library.model import RODSNet
from library.train import run_training


def demo_utils():
    """Demonstrates and verifies utility functions."""
    print("\n=== Demo: Utils ===")

    # 1. Test Metric Score Logic
    # Scenario A: Perfect prediction with confidence sigma=100
    # Metric = -ln(sqrt(2) * 100)
    y_true = np.array([2000.0])
    y_pred_mean = np.array([2000.0])
    y_pred_sigma = np.array([100.0])

    score = metric_score(y_true, y_pred_mean, y_pred_sigma)
    expected_score = -np.log(np.sqrt(2) * 100)
    print(
        f"Metric Score (Perfect Pred, Sigma=100): {score:.4f} (Expected: {expected_score:.4f})"
    )
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction"

    # Scenario B: Error=100, Sigma=50 (Clipped to 70)
    # Metric = - (sqrt(2)*100)/70 - ln(sqrt(2)*70)
    y_true = np.array([2000.0])
    y_pred_mean = np.array([2100.0])
    y_pred_sigma = np.array([50.0])

    score = metric_score(y_true, y_pred_mean, y_pred_sigma)
    term1 = -(np.sqrt(2) * 100) / 70
    term2 = -np.log(np.sqrt(2) * 70)
    expected_score_2 = term1 + term2
    print(
        f"Metric Score (Error=100, Sigma=50->70): {score:.4f} (Expected: {expected_score_2:.4f})"
    )
    assert np.isclose(
        score, expected_score_2, atol=1e-4
    ), "Metric calculation mismatch for clipped sigma"

    # 2. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)  # Sum=20, Count=2
    meter.update(20, n=2)  # Sum=60, Count=4
    print(f"AverageMeter: Avg={meter.avg} (Expected: 15.0)")
    assert meter.avg == 15.0, "AverageMeter logic incorrect"


def demo_loss():
    """Demonstrates and verifies the custom loss function."""
    print("\n=== Demo: Loss Function ===")
    criterion = LaplaceNLLLoss()

    # Create mock predictions (Batch Size = 2)
    # Preds: [Mean, Raw_Sigma]
    preds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    # Targets: [True_Value] (Standardized)
    targets = torch.tensor([0.0, 1.0])

    # Forward pass
    loss = criterion(preds, targets)
    print(f"Loss Value: {loss.item():.4f}")

    # Backward pass (Verify gradients)
    loss.backward()
    print("Gradients computed successfully.")
    assert preds.grad is not None, "Gradients were not computed during backward pass"


def demo_data_pipeline():
    """Demonstrates data loading and preprocessing."""
    print("\n=== Demo: Data Pipeline ===")

    # Modify Config for the demo to run quickly
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use main thread for simplicity in demo

    print("Initializing DataLoaders (Debug Mode)...")
    # debug=True loads a small subset of the data
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches:   {len(val_loader)}")

    # Fetch one batch to inspect
    imgs, tabular, targets = next(iter(train_loader))

    print(f"Batch Shapes:")
    print(f"  Images:  {imgs.shape} (Expected: [4, 3, 260, 260])")
    print(f"  Tabular: {tabular.shape} (Expected: [4, 5])")
    print(f"  Targets: {targets.shape} (Expected: [4, 1])")

    # Assertions
    assert imgs.shape == (4, 3, 260, 260), f"Image shape mismatch: {imgs.shape}"
    assert tabular.shape == (4, 5), f"Tabular shape mismatch: {tabular.shape}"
    assert targets.shape == (4, 1), f"Target shape mismatch: {targets.shape}"

    # Check Normalization (Images should be in [0, 1])
    print(f"Image Value Range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    assert (
        imgs.min() >= 0.0 and imgs.max() <= 1.0
    ), "Images are not properly normalized to [0, 1]"

    return imgs, tabular


def demo_model(imgs, tabular):
    """Demonstrates model instantiation and forward pass."""
    print("\n=== Demo: Model Architecture ===")

    # Initialize Model
    # We force CPU here for the architecture demo to ensure it runs anywhere
    device = torch.device("cpu")
    model = RODSNet().to(device)
    model.eval()

    print("RODSNet instantiated successfully.")

    # Run Forward Pass
    print("Running Forward Pass with sample batch...")
    with torch.no_grad():
        preds = model(imgs.to(device), tabular.to(device))

    print(f"Output Shape: {preds.shape} (Expected: [4, 2])")
    assert preds.shape == (4, 2), "Model output shape is incorrect"

    # Inspect outputs
    mean_pred = preds[:, 0]
    raw_sigma_pred = preds[:, 1]
    print(f"Sample Predictions (Mean): {mean_pred.numpy()}")
    print(f"Sample Predictions (Raw Sigma): {raw_sigma_pred.numpy()}")


def demo_training_loop():
    """Demonstrates the full training loop integration."""
    print("\n=== Demo: Training Loop ===")

    # Configure Config for a very short run
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.NUM_WORKERS = 0

    print(f"Starting training run (1 Epoch, Debug Subset)...")

    # Execute training
    # This uses the logic in library/train.py which handles optimizer, loss, and validation
    run_training(debug=True)

    print("Training run completed.")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Checkpoint found: {checkpoint_path} ({size_mb:.2f} MB)")
    else:
        print(
            "Note: No checkpoint saved (Validation score might not have improved in 1 epoch)."
        )


if __name__ == "__main__":
    # Set fixed seed
    seed_everything(42)

    # 1. Verify Utilities
    demo_utils()

    # 2. Verify Loss
    demo_loss()

    # 3. Verify Data Pipeline
    imgs, tabular = demo_data_pipeline()

    # 4. Verify Model
    demo_model(imgs, tabular)

    # 5. Verify Training Loop
    demo_training_loop()

    print("\nAll demonstrations passed successfully.")
