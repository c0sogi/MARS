import os
import sys
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_metric, LaplaceLogLikelihoodLoss
from library.data import LungDataset
from library.model import PAVENet
from library.engine import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides
    # We override specific config parameters to ensure the demo runs fast and offline.
    print("[1] Setting up configuration...")
    Config.setup()
    Config.BACKBONE_PRETRAINED = False  # Disable download to ensure offline execution
    Config.BATCH_SIZE = 2  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.EPOCHS = 1  # Run only 1 epoch

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration configured for demo mode.")

    # 2. Data Pipeline Demonstration
    print("\n[2] Verifying Data Pipeline...")

    # Load metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found at {train_meta_path}")

    df_full = pd.read_csv(train_meta_path)

    # Create a small subset (e.g., 6 samples) to ensure speed
    df_subset = df_full.head(6).copy()
    print(
        f"    Created subset of {len(df_subset)} samples from {len(df_full)} total records."
    )

    # Define transforms (same as in library/data.py)
    transforms = A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()])

    # Instantiate Dataset
    dataset = LungDataset(df_subset, mode="train", transform=transforms)

    # Instantiate DataLoader
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    # Verify Batch Structure
    batch = next(iter(loader))
    required_keys = [
        "image_axial",
        "image_coronal",
        "tabular",
        "anchor",
        "weeks",
        "target",
        "raw_base_fvc",
    ]
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key '{key}' in batch dictionary.")

    print("    Batch keys verified.")
    print(f"    Image Axial Shape: {batch['image_axial'].shape} (Expected: B, 3, H, W)")
    print(f"    Tabular Shape: {batch['tabular'].shape} (Expected: B, 7)")

    # Assert shapes
    B = batch["image_axial"].shape[0]
    assert batch["image_axial"].shape == (B, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert batch["image_coronal"].shape == (B, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert batch["tabular"].shape == (B, 7)
    assert batch["anchor"].shape == (B, 2)
    assert batch["weeks"].shape == (B, 1)

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (PAVENet)...")
    device = Config.DEVICE
    model = PAVENet().to(device)
    model.eval()

    # Move batch to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    anchor = batch["anchor"].to(device)
    weeks = batch["weeks"].to(device)
    raw_base_fvc = batch["raw_base_fvc"].to(device)

    # Forward Pass
    with torch.no_grad():
        fvc_pred, sigma_pred = model(
            img_ax, img_cor, tabular, anchor, weeks, raw_base_fvc
        )

    print(f"    Prediction FVC Shape: {fvc_pred.shape}")
    print(f"    Prediction Sigma Shape: {sigma_pred.shape}")

    assert fvc_pred.shape == (B, 1), "FVC prediction shape mismatch"
    assert sigma_pred.shape == (B, 1), "Sigma prediction shape mismatch"
    assert not torch.isnan(fvc_pred).any(), "Model produced NaN in FVC prediction"
    assert not torch.isnan(sigma_pred).any(), "Model produced NaN in Sigma prediction"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")
    criterion = LaplaceLogLikelihoodLoss()
    target = batch["target"].to(device)

    loss = criterion(fvc_pred, sigma_pred, target)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # 5. Engine / Trainer Verification
    print("\n[5] Verifying Trainer Loop...")
    trainer = Trainer()
    # Replace the internal model with our non-pretrained one to avoid re-initialization issues if any
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=Config.LR
    )  # Re-init optimizer for new model

    # Run one epoch of training on the subset loader
    print("    Running 'train_one_epoch' on subset...")
    train_loss = trainer.train_one_epoch(loader)
    print(f"    Subset Train Loss: {train_loss:.4f}")

    # Run validation on the subset loader
    print("    Running 'validate' on subset...")
    val_loss, val_metric = trainer.validate(loader)
    print(f"    Subset Val Loss: {val_loss:.4f} | Metric: {val_metric:.4f}")

    # 6. Metric Calculation Verification
    print("\n[6] Verifying Metric Calculation Logic...")
    # Synthetic data
    # Case 1: Perfect prediction
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    y_conf = np.array([100, 100])  # > 70

    # Metric formula: - (sqrt(2) * Delta) / Sigma - ln(sqrt(2) * Sigma)
    # Delta = 0
    # Term 1 = 0
    # Term 2 = ln(sqrt(2) * 100) = ln(141.42) approx 4.95
    # Metric = -4.95
    score = calculate_metric(y_true, y_pred, y_conf)
    print(f"    Perfect Prediction Score: {score:.4f}")

    # Check bounds
    expected_val = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score, expected_val, atol=1e-3
    ), "Metric calculation incorrect for perfect case"

    # Case 2: Large Error (clipped at 1000)
    y_true_bad = np.array([2000])
    y_pred_bad = np.array([4000])  # Error 2000 -> Clipped to 1000
    y_conf_bad = np.array([70])  # Clipped at 70 (min)

    score_bad = calculate_metric(y_true_bad, y_pred_bad, y_conf_bad)
    print(f"    Bad Prediction Score: {score_bad:.4f}")

    # Delta = 1000
    # Sigma = 70
    # Term 1 = (1.414 * 1000) / 70 = 20.2
    # Term 2 = ln(1.414 * 70) = ln(98.98) = 4.59
    # Metric = -20.2 - 4.59 = -24.79
    expected_bad = -(np.sqrt(2) * 1000) / 70 - np.log(np.sqrt(2) * 70)
    assert np.isclose(
        score_bad, expected_bad, atol=1e-3
    ), "Metric calculation incorrect for clipped error case"

    print("\n=== Demonstration Complete: All checks passed. ===")


if __name__ == "__main__":
    main()
