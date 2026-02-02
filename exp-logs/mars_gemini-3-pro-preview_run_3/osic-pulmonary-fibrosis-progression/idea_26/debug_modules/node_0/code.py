import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, MetricMonitor
from library.data import get_dataloaders, OSICDataset
from library.model import GCRNet
from library.train import train_one_epoch, evaluate
from library.predict import generate_submission


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.get_device()
    print(f"Device: {device}")

    # Ensure working directories exist (Config.setup() does this, but good to double check logic)
    assert os.path.exists(Config.CACHE_DIR), "Cache directory not created."
    assert os.path.exists(Config.CHECKPOINT_DIR), "Checkpoint directory not created."

    # 2. Data Loading Demonstration
    print("\n--- Demonstrating Data Loading ---")
    # We use the provided helper to get standard loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    images = batch["image"]
    tabular = batch["tabular"]
    target = batch["target"]
    patient_weeks = batch["patient_week"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image Shape: {images.shape}")
    print(f"Tabular Shape: {tabular.shape}")
    print(f"Target Shape: {target.shape}")

    # Assertions for Data
    # Image: (B, 3, H, W) -> (4, 3, 260, 260)
    assert images.shape == (
        4,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {images.shape}"
    # Tabular: (B, 6) -> [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke]
    assert tabular.shape == (4, 6), f"Unexpected tabular shape: {tabular.shape}"
    # Target: (B,) or (B, 1)
    assert target.numel() == 4, "Target batch size mismatch"

    # 3. Model Instantiation & Forward Pass
    print("\n--- Demonstrating Model Architecture ---")
    model = GCRNet().to(device)

    # Move data to device
    images = images.to(device)
    tabular = tabular.to(device)
    target = target.to(device)

    # Forward pass
    preds = model(images, tabular)
    print(f"Prediction Shape: {preds.shape}")

    # Assertions for Model
    # Output should be (B, 2) -> [Mu, Sigma]
    assert preds.shape == (4, 2), f"Unexpected prediction shape: {preds.shape}"

    # Check Sigma positivity (Constraint in model: Softplus + epsilon)
    sigmas = preds[:, 1]
    assert torch.all(
        sigmas > 0
    ), "Model produced non-positive confidence values (Sigma)."

    # 4. Loss Function Verification
    print("\n--- Demonstrating Loss Function ---")
    criterion = LaplaceLogLikelihoodLoss()
    loss = criterion(preds, target)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"

    # 5. Metric Monitor Verification
    print("\n--- Demonstrating Metric Monitor ---")
    monitor = MetricMonitor()
    monitor.update(loss.item(), preds, target)

    avg_loss = monitor.get_avg_loss()
    avg_score = monitor.get_avg_score()
    print(f"Monitor Avg Loss: {avg_loss}")
    print(f"Monitor Avg Score: {avg_score}")

    assert avg_loss == loss.item(), "Monitor loss tracking incorrect"

    # 6. Training Loop Demonstration (Optimized for Speed)
    print("\n--- Demonstrating Training Step (Subset) ---")
    # Create a tiny subset to simulate an epoch quickly
    full_train_ds = OSICDataset(mode="train")
    subset_indices = list(range(4))  # Only 4 samples
    small_ds = Subset(full_train_ds, subset_indices)
    small_loader = DataLoader(small_ds, batch_size=2, shuffle=False)

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch on the small subset
    train_loss, train_score = train_one_epoch(
        model, small_loader, criterion, optimizer, device
    )
    print(f"Subset Train Loss: {train_loss:.4f}, Score: {train_score:.4f}")

    # Run evaluation on the small subset
    val_loss, val_score = evaluate(model, small_loader, criterion, device)
    print(f"Subset Val Loss: {val_loss:.4f}, Score: {val_score:.4f}")

    # 7. Inference & Submission Demonstration
    print("\n--- Demonstrating Inference & Submission ---")

    # Save the current model as 'best_model.pth' so generate_submission can find it
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"Dummy checkpoint saved to {best_model_path}")

    # Generate submission
    # This uses the full test set defined in metadata/test.csv and sample_submission.csv
    # The test set is small enough to run quickly.
    generate_submission()

    # Verify Output
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head(3))

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Check if values are populated
    assert not sub_df["FVC"].isnull().any(), "Submission contains null FVCs"
    assert (
        not sub_df["Confidence"].isnull().any()
    ), "Submission contains null Confidences"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
