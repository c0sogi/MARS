import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import OSICDataset, get_transforms
from library.model import LARFNet
from library.train import LaplaceLoss


def run_demo():
    print("Initializing Demo...")

    # 1. Runtime Configuration Modification for Speed
    # We modify the Config class attributes directly to run a lightweight version
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Only use 10 samples for demo
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.DEVICE = (
        "cpu"  # Force CPU for simple demo stability, or use cuda if preferred
    )
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    print(f"Running on device: {Config.DEVICE}")
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Dataset and DataLoader Demonstration
    # ==========================================
    print("\n--- 1. Dataset & DataLoader Verification ---")

    # Initialize Dataset (Training Mode)
    train_transform = get_transforms(mode="train")
    train_dataset = OSICDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=train_transform
    )

    # Apply Debug Subset logic manually for the demo script
    indices = list(range(min(len(train_dataset), Config.DEBUG_SUBSET_SIZE)))
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    print(f"Dataset initialized. Subset size: {len(train_subset)}")

    # Verify single item structure
    sample = train_subset[0]
    required_keys = [
        "image_axial",
        "image_coronal",
        "tabular",
        "target",
        "week",
        "baseline_fvc",
        "patient_week",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    print("Sample keys verified.")
    print(f"Image Shape: {sample['image_axial'].shape}")  # Should be (3, 224, 224)
    print(f"Tabular Shape: {sample['tabular'].shape}")  # Should be (7,)

    # Initialize DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Shapes
    img_axial = batch["image_axial"]
    img_coronal = batch["image_coronal"]
    tabular = batch["tabular"]
    target = batch["target"]

    assert img_axial.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected Axial shape: {img_axial.shape}"
    assert img_coronal.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected Coronal shape: {img_coronal.shape}"
    assert tabular.shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Unexpected Tabular shape: {tabular.shape}"
    assert target.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected Target shape: {target.shape}"

    print("Batch shapes verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- 2. Model Initialization & Forward Pass ---")

    model = LARFNet().to(Config.DEVICE)
    model.eval()  # Set to eval for deterministic check first

    # Move batch to device
    img_axial = img_axial.to(Config.DEVICE)
    img_coronal = img_coronal.to(Config.DEVICE)
    tabular = tabular.to(Config.DEVICE)
    week = batch["week"].to(Config.DEVICE)
    baseline_fvc = batch["baseline_fvc"].to(Config.DEVICE)

    # Forward Pass with trajectory inputs (week, baseline)
    # This should return (fvc_pred, sigma_pred)
    with torch.no_grad():
        fvc_pred, sigma_pred = model(
            img_axial, img_coronal, tabular, week=week, baseline_fvc=baseline_fvc
        )

    print(f"Prediction FVC: {fvc_pred.cpu().numpy()}")
    print(f"Prediction Sigma: {sigma_pred.cpu().numpy()}")

    assert fvc_pred.shape == (Config.BATCH_SIZE,), "FVC prediction shape mismatch"
    assert sigma_pred.shape == (Config.BATCH_SIZE,), "Sigma prediction shape mismatch"
    assert torch.all(
        sigma_pred >= 0
    ), "Sigma predictions must be non-negative (Softplus)"

    print("Forward pass successful.")

    # ==========================================
    # 4. Loss and Metric Calculation
    # ==========================================
    print("\n--- 3. Loss & Metric Calculation ---")

    criterion = LaplaceLoss()
    target = target.to(Config.DEVICE)

    # Calculate Loss
    loss = criterion(target, fvc_pred, sigma_pred)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # Calculate Metric using Utility
    # We use numpy arrays for the utility function
    metric_val = calculate_metric(
        target.cpu().numpy(), fvc_pred.cpu().numpy(), sigma_pred.cpu().numpy()
    )
    print(f"Calculated Metric: {metric_val:.4f}")

    # Sanity check: Metric should be negative
    assert metric_val < 0, "Metric should be negative (Log Likelihood)"

    # ==========================================
    # 5. Training Step Simulation
    # ==========================================
    print("\n--- 4. Training Step Simulation ---")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Zero grad
    optimizer.zero_grad()

    # Forward
    fvc_train, sigma_train = model(
        img_axial, img_coronal, tabular, week=week, baseline_fvc=baseline_fvc
    )

    # Loss
    train_loss = criterion(target, fvc_train, sigma_train)

    # Backward
    train_loss.backward()

    # Check gradients exist
    assert model.head[0].weight.grad is not None, "Gradients not computed for head"

    # Step
    optimizer.step()
    print("Optimizer step executed successfully.")

    # ==========================================
    # 6. Inference / Test Set Demonstration
    # ==========================================
    print("\n--- 5. Inference Demonstration ---")

    test_dataset = OSICDataset(
        csv_path=Config.TEST_CSV, mode="test", transform=get_transforms(mode="val")
    )

    # Just take one sample
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_batch = next(iter(test_loader))

        t_img_ax = test_batch["image_axial"].to(Config.DEVICE)
        t_img_cor = test_batch["image_coronal"].to(Config.DEVICE)
        t_tab = test_batch["tabular"].to(Config.DEVICE)
        t_week = test_batch["week"].to(Config.DEVICE)
        t_base = test_batch["baseline_fvc"].to(Config.DEVICE)

        model.eval()
        with torch.no_grad():
            pred_fvc, pred_conf = model(
                t_img_ax, t_img_cor, t_tab, week=t_week, baseline_fvc=t_base
            )

        print(f"Test Sample ID: {test_batch['patient_week'][0]}")
        print(f"Predicted FVC: {pred_fvc.item():.2f}")
        print(f"Predicted Confidence: {pred_conf.item():.2f}")
    else:
        print("Test dataset is empty (check metadata generation).")

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
