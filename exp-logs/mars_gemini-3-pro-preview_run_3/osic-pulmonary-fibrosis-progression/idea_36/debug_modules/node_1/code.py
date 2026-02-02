import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import UCOSRNet
from library.train import LaplaceNLLLoss, train_fn, eval_fn


def main():
    print("Starting UCOSR-Net Library Demonstration...")

    # 1. Setup & Reproducibility
    seed_everything(Config.SEED)

    # 2. Optimize Config for Speed (Demo Mode)
    print("Configuring for fast demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Create Data Subsets
    # We use a small subset of the metadata to ensure the code runs in seconds.
    print("Creating data subsets...")
    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/val.csv"

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata files not found. Ensure ./metadata exists.")

    train_df = pd.read_csv(train_meta_path)
    val_df = pd.read_csv(val_meta_path)

    # Take top N rows
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")

    train_df.head(12).to_csv(subset_train_path, index=False)
    val_df.head(8).to_csv(subset_val_path, index=False)

    # Update Config to use subsets
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path

    # 4. Verify Data Loading
    print("Verifying Data Loading...")
    train_loader, val_loader = get_dataloaders(
        Config.TRAIN_CSV, Config.VAL_CSV, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Assertions for Data
    assert "image" in batch
    assert "tabular" in batch
    assert "target" in batch
    assert "patient_week" in batch

    # Check shapes
    # Image: (Batch, 3, H, W) -> (4, 3, 260, 260) based on Config
    assert batch["image"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {batch['image'].shape}"
    # Tabular: (Batch, 5)
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        Config.TABULAR_INPUT_DIM,
    ), f"Tabular shape mismatch: {batch['tabular'].shape}"
    # Target: (Batch,)
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch: {batch['target'].shape}"

    print("Data Loading Verified.")

    # 5. Verify Model Architecture
    print("Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = UCOSRNet().to(device)

    imgs = batch["image"].to(device)
    tabular = batch["tabular"].to(device)

    # Forward Pass
    mu, sigma = model(imgs, tabular)

    # Assertions for Model Output
    assert mu.shape == (Config.BATCH_SIZE,), "Output mu shape mismatch"
    assert sigma.shape == (Config.BATCH_SIZE,), "Output sigma shape mismatch"
    assert torch.all(sigma > 0), "Sigma must be strictly positive (Softplus)"

    print("Model Architecture Verified.")

    # 6. Verify Metric Logic
    print("Verifying Metric Calculation...")
    # Test Case: Perfect prediction with high confidence
    # Formula: - (sqrt(2) * Delta) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped)
    # If Delta=0, Sigma=100 (clipped=100)
    # Metric = 0 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma_pred = np.array([100.0])

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma_pred)
    expected_score = -np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric calculation mismatch. Got {score}, expected {expected_score}"

    # Test Case: Clipping
    # Sigma=10 (clipped to 70)
    sigma_small = np.array([10.0])
    score_clipped = laplace_log_likelihood_metric(y_true, y_pred, sigma_small)
    expected_clipped = -np.log(np.sqrt(2) * 70)

    assert np.isclose(
        score_clipped, expected_clipped, atol=1e-4
    ), f"Metric clipping mismatch. Got {score_clipped}, expected {expected_clipped}"

    print("Metric Logic Verified.")

    # 7. Verify Loss Function and Training Step
    print("Verifying Training Step...")
    criterion = LaplaceNLLLoss()

    # Calculate initial loss
    targets = batch["target"].to(device)
    loss = criterion((mu, sigma), targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() != 0, "Loss is zero (unlikely for initialized model)"

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one training epoch
    print(f"Running 1 epoch on {len(train_loader)} batches...")
    train_loss = train_fn(train_loader, model, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Run evaluation
    print(f"Running evaluation on {len(val_loader)} batches...")
    val_loss, val_metric = eval_fn(val_loader, model, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

    assert train_loss != 0
    # Metric is usually negative, ensure it's a valid float
    assert isinstance(val_metric, float)

    print("Training Loop Verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
