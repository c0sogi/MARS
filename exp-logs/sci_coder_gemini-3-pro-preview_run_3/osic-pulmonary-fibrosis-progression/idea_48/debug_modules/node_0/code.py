import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, inverse_transform, calculate_metric
from library.data import get_dataloaders
from library.model import CASDSNet
from library.train import LLLLoss, train_epoch, val_epoch, generate_submission


def main():
    print("=== Starting OSIC Pulmonary Fibrosis Demo ===")

    # 1. Setup & Configuration Override
    # We override Config attributes to ensure the demo runs quickly (Debug mode)
    # and writes to a separate directory.
    seed_everything(42)

    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo stability

    # Define a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Clean and recreate directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Pipeline Verification
    print("\n[Step 1/5] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        load_cached_data=False,  # Force processing to verify logic
    )

    # Fetch one batch
    try:
        images, tabular, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"   Image Batch Shape: {images.shape}")  # Expected: (Batch, 3, 260, 260)
    print(f"   Tabular Batch Shape: {tabular.shape}")  # Expected: (Batch, 5)
    print(f"   Target Batch Shape: {targets.shape}")  # Expected: (Batch, 1)

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Image tensor shape"
    assert tabular.shape == (Config.BATCH_SIZE, 5), "Incorrect Tabular tensor shape"
    assert targets.shape == (Config.BATCH_SIZE, 1), "Incorrect Target tensor shape"
    print("   -> Data Loading successful.")

    # 3. Model Architecture Verification
    print("\n[Step 2/5] Verifying Model Architecture...")
    model = CASDSNet().to(device)

    # Move data to device
    images = images.to(device)
    tabular = tabular.to(device)
    targets = targets.to(device).squeeze(-1)  # Squeeze to (Batch,) for loss calculation

    # Forward pass
    mu, sigma = model(images, tabular)

    print(f"   Output Mu Shape: {mu.shape}")
    print(f"   Output Sigma Shape: {sigma.shape}")

    # Assertions
    assert mu.shape == (Config.BATCH_SIZE,), "Mu output shape mismatch"
    assert sigma.shape == (Config.BATCH_SIZE,), "Sigma output shape mismatch"
    assert (sigma > 0).all(), "Sigma must be positive (Softplus constraint failed)"
    print("   -> Model Forward Pass successful.")

    # 4. Loss & Metric Verification
    print("\n[Step 3/5] Verifying Loss and Metric...")
    criterion = LLLLoss()

    # Calculate Loss
    loss = criterion(mu, sigma, targets)
    print(f"   Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss is not a scalar"

    # Calculate Metric (requires inverse transform)
    mu_orig, sigma_orig = inverse_transform(mu, sigma)
    target_orig = targets.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN

    metric_score = calculate_metric(target_orig, mu_orig, sigma_orig)
    print(f"   Calculated Metric Score: {metric_score:.4f}")
    assert isinstance(metric_score, float), "Metric is not a float"
    print("   -> Loss and Metric calculation successful.")

    # 5. Training Loop Simulation
    print("\n[Step 4/5] Simulating Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train Epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"   Epoch 1 Train Loss: {train_loss:.4f}")

    # Val Epoch
    val_loss, val_score = val_epoch(model, val_loader, criterion, device)
    print(f"   Epoch 1 Val Loss: {val_loss:.4f} | Score: {val_score:.4f}")

    # Save checkpoint (simulating best model save)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file not created"
    print("   -> Training loop simulation successful.")

    # 6. Inference & Submission
    print("\n[Step 5/5] Generating Submission...")
    # generate_submission uses the model passed to it, and internally loads test data
    generate_submission(model, device)

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    sub_df = pd.read_csv(sub_path)
    print(f"   Submission Rows: {len(sub_df)}")
    print(f"   Columns: {list(sub_df.columns)}")

    assert (
        "FVC" in sub_df.columns and "Confidence" in sub_df.columns
    ), "Missing required columns in submission"
    assert len(sub_df) > 0, "Submission file is empty"
    print("   -> Submission generation successful.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
