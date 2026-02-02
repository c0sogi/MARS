import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import LungDataset
from library.model import DPSDAN
from library.loss import LaplaceLogLikelihoodLoss
from library.train import set_seed


def demo_main():
    print("Starting DP-SDAN Library Demonstration...")

    # 1. Setup and Configuration Overrides for Demo Speed
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override Config for rapid execution
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("\n[1] Verifying Data Loading...")
    # Instantiate Datasets
    train_ds = LungDataset(mode="train")
    val_ds = LungDataset(mode="val")
    test_ds = LungDataset(mode="test")

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Test samples:  {len(test_ds)}")

    # Create a DataLoader and fetch one batch
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = [
        "img_axial",
        "img_coronal",
        "tab_dense",
        "baseline_fvc",
        "delta_week",
        "target_fvc",
        "patient_id",
    ]
    for k in expected_keys:
        assert k in batch, f"Missing key {k} in batch"

    # Verify Shapes
    # Image: (Batch, 3, 224, 224)
    assert batch["img_axial"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {batch['img_axial'].shape}"
    assert batch["img_coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image Shape: {batch['img_coronal'].shape}"

    # Tabular Dense: (Batch, 6)
    assert batch["tab_dense"].shape == (
        Config.BATCH_SIZE,
        Config.TABULAR_INPUT_DIM,
    ), f"Incorrect Tabular Shape: {batch['tab_dense'].shape}"

    print("  Data Loading verification successful.")

    print("\n[2] Verifying Model Architecture...")
    model = DPSDAN().to(device)
    model.train()

    # Move batch to device
    img_axial = batch["img_axial"].to(device)
    img_coronal = batch["img_coronal"].to(device)
    tab_dense = batch["tab_dense"].to(device)

    baseline_fvc = batch["baseline_fvc"].to(device)
    delta_week = batch["delta_week"].to(device)
    target_fvc = batch["target_fvc"].to(device)

    # Forward Pass
    alpha, sigma_base, sigma_growth = model(img_axial, img_coronal, tab_dense)

    # Verify Output Shapes: (Batch,)
    assert alpha.shape == (Config.BATCH_SIZE,), "Alpha output shape mismatch"
    assert sigma_base.shape == (Config.BATCH_SIZE,), "Sigma_base output shape mismatch"
    assert sigma_growth.shape == (
        Config.BATCH_SIZE,
    ), "Sigma_growth output shape mismatch"

    # Verify Constraints (Sigma must be positive due to Softplus)
    assert (sigma_base > 0).all(), "Sigma base must be positive"
    assert (sigma_growth > 0).all(), "Sigma growth must be positive"

    print("  Model Forward Pass verification successful.")

    print("\n[3] Verifying Loss Computation and Optimization...")
    loss_fn = LaplaceLogLikelihoodLoss()

    # Calculate Loss
    loss = loss_fn(
        alpha, sigma_base, sigma_growth, baseline_fvc, delta_week, target_fvc
    )

    # Check Loss Validity
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    print(f"  Computed Loss: {loss.item():.4f}")

    # Optimization Step
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Check if gradients are computed (check one parameter)
    # Access the first layer of the tabular expander
    param = next(model.tabular_expander.parameters())
    assert param.grad is not None, "Gradients not computed"

    optimizer.step()
    print("  Backward pass and Optimizer step successful.")

    print("\n[4] Verifying Inference Logic...")
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_batch = next(iter(test_loader))

    with torch.no_grad():
        t_img_ax = test_batch["img_axial"].to(device)
        t_img_cor = test_batch["img_coronal"].to(device)
        t_tab = test_batch["tab_dense"].to(device)

        t_base_fvc = test_batch["baseline_fvc"].to(device)
        t_delta = test_batch["delta_week"].to(device)

        # Forward
        t_alpha, t_sigma_base, t_sigma_growth = model(t_img_ax, t_img_cor, t_tab)

        # Reconstruct Prediction
        pred_fvc = t_base_fvc + t_alpha * t_delta
        pred_sigma = t_sigma_base + t_sigma_growth * torch.abs(t_delta)

        # Check values
        pred_fvc_np = pred_fvc.cpu().numpy()
        pred_sigma_np = pred_sigma.cpu().numpy()

        print(f"  Sample Prediction (FVC): {pred_fvc_np[0]:.2f}")
        print(f"  Sample Confidence (Sigma): {pred_sigma_np[0]:.2f}")

        assert len(pred_fvc_np) == Config.BATCH_SIZE
        assert len(pred_sigma_np) == Config.BATCH_SIZE

    print("  Inference logic verification successful.")

    print("\n" + "=" * 40)
    print("DP-SDAN DEMONSTRATION COMPLETE")
    print("=" * 40)


if __name__ == "__main__":
    demo_main()
