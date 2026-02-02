import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config, setup_reproducibility
from library.data import LungDataset, get_transforms
from library.model import CalibratedSymmetricDualAxisNetwork
from library.train import LaplaceLoss, train_epoch
from library.utils import calculate_metric


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    setup_reproducibility(42)

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Small subset for speed
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Use a specific cache dir for this demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Cache Directory: {Config.CACHE_DIR}")

    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"    Loaded {len(train_df)} training samples.")

    # Instantiate Dataset
    dataset = LungDataset(
        train_df,
        cache_dir=Config.CACHE_DIR,
        transform=get_transforms("train"),
        mode="train",
    )

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Fetch one batch
    batch = next(iter(loader))

    # Verify Batch Structure
    imgs_ax = batch["image_axial"]
    imgs_cor = batch["image_coronal"]
    tabular = batch["tabular"]
    targets = batch["target"]
    metadata = batch["metadata"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(
        f"    Axial Image Shape: {imgs_ax.shape} (Expected: [{Config.BATCH_SIZE}, 3, 224, 224])"
    )
    print(
        f"    Coronal Image Shape: {imgs_cor.shape} (Expected: [{Config.BATCH_SIZE}, 3, 224, 224])"
    )
    print(f"    Tabular Shape: {tabular.shape} (Expected: [{Config.BATCH_SIZE}, 7])")
    print(f"    Target Shape: {targets.shape} (Expected: [{Config.BATCH_SIZE}])")

    # Assertions
    assert imgs_ax.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Axial Image Shape"
    assert imgs_cor.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Coronal Image Shape"
    assert tabular.shape == (Config.BATCH_SIZE, 7), "Incorrect Tabular Feature Shape"

    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = CalibratedSymmetricDualAxisNetwork().to(device)
    model.eval()

    # Move batch to device
    imgs_ax = imgs_ax.to(device)
    imgs_cor = imgs_cor.to(device)
    tabular = tabular.to(device)

    # Forward Pass
    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

    print(f"    Alpha Output Shape: {alpha.shape}")
    print(f"    Sigma Base Output Shape: {sigma_base.shape}")
    print(f"    Sigma Growth Output Shape: {sigma_growth.shape}")

    # Assertions
    assert alpha.shape == (Config.BATCH_SIZE, 1), "Alpha shape mismatch"
    assert sigma_base.shape == (Config.BATCH_SIZE, 1), "Sigma Base shape mismatch"
    assert sigma_growth.shape == (Config.BATCH_SIZE, 1), "Sigma Growth shape mismatch"

    # Check positivity constraint on sigmas (Softplus)
    assert (sigma_base >= 0).all(), "Sigma Base must be positive"
    assert (sigma_growth >= 0).all(), "Sigma Growth must be positive"

    # 4. Metric and Loss Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Metric and Loss Logic...")

    # Test Metric Calculation
    # Scenario: Perfect prediction with confidence 100
    # Metric = - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100) = -ln(141.42) ~= -4.95
    y_true_dummy = np.array([2000.0])
    y_pred_dummy = np.array([2000.0])
    sigma_dummy = np.array([100.0])

    metric_val = calculate_metric(y_true_dummy, y_pred_dummy, sigma_dummy)
    expected_val = -np.log(np.sqrt(2) * 100)
    print(f"    Calculated Metric: {metric_val:.4f}")
    print(f"    Expected Metric:   {expected_val:.4f}")

    assert np.isclose(
        metric_val, expected_val, atol=1e-4
    ), "Metric calculation mismatch"

    # Test Loss Function
    criterion = LaplaceLoss()

    # Create tensors for loss
    t_true = torch.tensor([2000.0], device=device)
    t_pred = torch.tensor([2100.0], device=device)  # Error 100
    t_sigma = torch.tensor([70.0], device=device)  # Min clipped sigma

    # Manual calc:
    # Delta = 100. Sigma = 70.
    # Term1 = (1.414 * 100) / 70 = 2.02
    # Term2 = ln(1.414 * 70) = ln(98.99) = 4.595
    # Total = 6.615
    loss_val = criterion(t_true, t_pred, t_sigma)
    print(f"    Calculated Loss: {loss_val.item():.4f}")

    assert not torch.isnan(loss_val), "Loss resulted in NaN"
    assert loss_val.item() > 0, "Loss should be positive"

    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch using the library function
    # This verifies the integration of data loading, model forward, and backward pass
    avg_loss = train_epoch(model, loader, optimizer, criterion, device)

    print(f"    Training Step Complete. Average Loss: {avg_loss:.4f}")
    assert avg_loss > 0, "Training loss should be positive"

    # 6. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Inference (Test Mode)...")

    test_df = pd.read_csv(Config.TEST_META_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    test_dataset = LungDataset(
        test_df,
        cache_dir=Config.CACHE_DIR,
        transform=get_transforms("test"),
        mode="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    batch = next(iter(test_loader))

    imgs_ax = batch["image_axial"].to(device)
    imgs_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)

    # Metadata for reconstruction
    weeks = batch["metadata"]["Weeks"].to(device)
    base_weeks = batch["metadata"]["Baseline_Week"].to(device)
    base_fvc = batch["metadata"]["Baseline_FVC"].to(device)

    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

        # Reconstruct Prediction (Linear Model)
        dt = weeks - base_weeks
        fvc_pred = base_fvc + alpha.view(-1) * dt
        confidence = sigma_base.view(-1) + sigma_growth.view(-1) * torch.abs(dt)

    print(f"    Input Weeks: {weeks.cpu().numpy()}")
    print(f"    Predicted FVC: {fvc_pred.cpu().numpy()}")
    print(f"    Predicted Confidence: {confidence.cpu().numpy()}")

    assert len(fvc_pred) == Config.BATCH_SIZE, "Inference output size mismatch"

    print("\n=== Demonstration Complete: All checks passed ===")


if __name__ == "__main__":
    main()
