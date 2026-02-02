import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders, LungDataset
from library.model import CIDSNet
from library.train import LaplaceLoss, run_training, validate


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up demonstration configuration...")

    # Override Config for speed and isolation
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Limit data and training duration for the demo
    Config.MAX_TRAIN_SAMPLES = 20  # Use only 20 samples for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading Pipeline...")

    # Load dataloaders
    train_loader, val_loader, test_loader, processor = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Check if processor is fitted
    assert (
        processor.fitted
    ), "DataProcessor should be fitted after calling get_dataloaders"
    assert processor.target_mean is not None, "Target mean should be calculated"
    assert processor.target_std is not None, "Target std should be calculated"

    # Fetch one batch from training loader
    images, tabular, targets = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Tabular Shape: {tabular.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    # Validate Shapes
    # Images: (B, 3, 260, 260)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"

    # Tabular: (B, 5) -> [Base_FVC, Time, Age, Sex, Smoke]
    assert tabular.shape == (
        Config.BATCH_SIZE,
        5,
    ), f"Incorrect tabular shape: {tabular.shape}"

    # Targets: (B,)
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect targets shape: {targets.shape}"

    print("    Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    model = CIDSNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    outputs = model(images, tabular)

    print(f"    Model Output Shape: {outputs.shape}")

    # Validate Output Shape: (B, 2) -> [Mean, Sigma]
    assert outputs.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Incorrect model output shape: {outputs.shape}"

    # Check for NaN/Inf
    assert torch.isfinite(outputs).all(), "Model output contains NaNs or Infs"

    # Verify Sigma is positive (Softplus + epsilon used in model)
    sigmas = outputs[:, 1]
    assert (sigmas > 0).all(), "Predicted sigma values must be positive"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Laplace Loss...")

    criterion = LaplaceLoss()
    targets = targets.to(device)

    loss = criterion(outputs, targets)

    print(f"    Calculated Loss: {loss.item():.4f}")

    assert torch.isfinite(loss), "Loss value is not finite"
    assert loss.item() > -100, "Loss value is suspiciously low (check log term)"

    print("    Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 5. Metric Calculation Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Competition Metric (Score Function)...")

    # Manual Test Case
    # True FVC: 2000
    # Pred FVC: 2000 (Delta = 0)
    # Pred Sigma: 100 (Clipped = 100)
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100)
    #        = 0 - ln(141.42) ~= -4.9517

    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])

    metric = score_function(y_true, y_pred, sigma)
    expected_metric = -np.log(np.sqrt(2) * 100)

    print(f"    Calculated Metric: {metric:.4f}")
    print(f"    Expected Metric:   {expected_metric:.4f}")

    assert np.isclose(
        metric, expected_metric, atol=1e-4
    ), f"Metric calculation mismatch. Got {metric}, expected {expected_metric}"

    print("    Metric verification passed.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Loop (1 Epoch, Subset of Data)...")

    # run_training handles initialization, training, validation, and checkpointing
    best_metric = run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    print(f"    Training loop completed. Best Metric: {best_metric:.4f}")
    print(f"    Checkpoint saved at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 7. Inference / Submission Generation
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get scaler stats for inverse transform
    target_mean = processor.target_mean
    target_std = processor.target_std

    predictions = []

    with torch.no_grad():
        for images, tabular, _ in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # Forward
            out = model(images, tabular)

            # Inverse Scale
            pred_fvc = out[:, 0].cpu().numpy() * target_std + target_mean
            pred_sigma = out[:, 1].cpu().numpy() * target_std

            # Store (just for verification)
            batch_preds = np.stack([pred_fvc, pred_sigma], axis=1)
            predictions.append(batch_preds)

    all_preds = np.concatenate(predictions, axis=0)

    print(f"    Total Predictions: {len(all_preds)}")
    print(f"    Prediction Sample (FVC, Sigma): {all_preds[0]}")

    # Check against sample submission length in metadata
    # Note: test_loader is based on sample_submission.csv rows
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    assert len(all_preds) == len(
        sample_sub
    ), f"Number of predictions ({len(all_preds)}) does not match sample submission ({len(sample_sub)})"

    print("    Inference verification passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
