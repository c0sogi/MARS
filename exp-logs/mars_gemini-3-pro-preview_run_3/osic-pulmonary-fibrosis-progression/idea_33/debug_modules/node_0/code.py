import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import ZIOSRNet
from library.train import LaplaceNLLLoss, train_one_epoch, validate, generate_submission


def run_demo():
    print("=== Starting Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")
    # Override Config parameters to make the run faster
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.BATCH_SIZE = 8  # Smaller batch size for quick iterations

    # Ensure working directory exists (as per Config)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Epochs set to: {Config.EPOCHS}")
    print(f"    Batch Size set to: {Config.BATCH_SIZE}")
    print("    Configuration updated.\n")

    # ---------------------------------------------------------
    # 2. Utility Verification
    # ---------------------------------------------------------
    print("[2] Verifying Utility Functions...")
    seed_everything(Config.SEED)
    print("    Seed set successfully.")

    # Test compute_metric with known values
    # Case 1: Perfect prediction (Delta=0), Confidence=70 (clipped min)
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    true_fvc = np.array([2000.0])
    pred_fvc = np.array([2000.0])
    pred_sigma = np.array([70.0])

    metric_val = compute_metric(true_fvc, pred_fvc, pred_sigma)
    expected_val = -np.log(np.sqrt(2) * 70)

    print(f"    Metric Check: Calculated={metric_val:.4f}, Expected={expected_val:.4f}")
    assert np.isclose(
        metric_val, expected_val, atol=1e-4
    ), "Metric calculation mismatch!"
    print("    Utility verification passed.\n")

    # ---------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # ---------------------------------------------------------
    print("[3] Initializing Data Loaders...")
    # This will trigger image caching (or loading from cache) and dataset creation
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify shapes
    images, features, targets = next(iter(train_loader))

    print("    Batch Shapes:")
    print(f"      Images:   {images.shape} (Expected: B, 3, 260, 260)")
    print(f"      Features: {features.shape} (Expected: B, 5)")
    print(f"      Targets:  {targets.shape} (Expected: B)")

    # Assertions
    assert images.ndim == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (slices)"
    assert features.shape[1] == 5, "Features should have 5 clinical inputs"
    assert targets.ndim == 1, "Targets should be 1D tensor"
    print("    Data pipeline verification passed.\n")

    # ---------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("[4] Initializing ZIOSRNet Model...")
    device = Config.DEVICE
    model = ZIOSRNet().to(device)

    # Move batch to device
    images = images.to(device)
    features = features.to(device)

    print("    Performing Forward Pass...")
    mu_pred, sigma_pred = model(images, features)

    print(f"    Output Shapes: Mu={mu_pred.shape}, Sigma={sigma_pred.shape}")

    # Assertions
    assert mu_pred.shape == targets.shape, "Mu prediction shape mismatch"
    assert sigma_pred.shape == targets.shape, "Sigma prediction shape mismatch"
    assert torch.all(sigma_pred > 0), "Sigma predictions must be positive (Softplus)"
    print("    Model verification passed.\n")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("[5] Running Training Loop (1 Epoch)...")
    criterion = LaplaceNLLLoss()

    # Setup simple optimizer for demo
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run training for one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Training complete. Epoch Loss: {train_loss:.6f}")

    assert not np.isnan(train_loss), "Training loss returned NaN"
    print("    Training loop verification passed.\n")

    # ---------------------------------------------------------
    # 6. Validation & Inference
    # ---------------------------------------------------------
    print("[6] Running Validation...")
    # Retrieve stats from dataset for inverse transform inside validate/generate
    stats = train_loader.dataset.stats

    val_metric = validate(model, val_loader, device, stats)
    print(f"    Validation Metric: {val_metric:.6f}")
    assert isinstance(val_metric, float), "Validation metric should be a float"

    print("    Generating Submission...")
    generate_submission(model, test_loader, device, stats)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"    Submission File: {Config.SUBMISSION_PATH}")
    print(f"    Rows: {len(sub_df)}")
    print(f"    Columns: {list(sub_df.columns)}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if Confidence is clipped correctly (>= 70)
    min_conf = sub_df["Confidence"].min()
    print(f"    Min Confidence in Submission: {min_conf}")
    assert min_conf >= 70, "Confidence values must be >= 70"

    print("    Inference verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
