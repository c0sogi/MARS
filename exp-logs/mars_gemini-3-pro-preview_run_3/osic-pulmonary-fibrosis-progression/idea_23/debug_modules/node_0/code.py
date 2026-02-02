import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import (
    OSICDataset,
    get_transforms,
    preprocess_tabular,
    add_baseline_features,
    get_dataloaders,
)
from library.model import MCDSRNet, metric_aligned_loss, train_one_epoch, validate


def main():
    print("=== Starting Demonstration of Pulmonary Fibrosis Library ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up Configuration...")
    seed_everything(Config.SEED)

    # Override Config for rapid demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist (Config does this, but good to double check context)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"    Loaded train metadata with {len(train_df)} rows.")

    # Test Feature Engineering
    print("    Testing feature engineering functions...")
    # Take a small subset for manual verification
    demo_df = train_df.head(10).copy()

    # Add baseline features (base_FVC, base_Percent, t_rel)
    demo_df = add_baseline_features(demo_df)
    assert (
        "base_FVC" in demo_df.columns
    ), "base_FVC column missing after add_baseline_features"
    assert (
        "t_rel" in demo_df.columns
    ), "t_rel column missing after add_baseline_features"

    # Preprocess tabular (scaling/encoding)
    demo_df, stats = preprocess_tabular(demo_df)
    assert "Sex_encoded" in demo_df.columns, "Sex_encoded missing"
    assert "Age_scaled" in demo_df.columns, "Age_scaled missing"

    print("    Feature engineering successful.")

    # Test Dataset Class
    print("    Testing OSICDataset...")
    dataset = OSICDataset(
        demo_df,
        Config.TRAIN_IMG_DIR,
        mode="train",
        transform=get_transforms("train"),
        cache_dir=Config.CACHE_DIR,
    )

    # Fetch one sample
    sample = dataset[0]
    image = sample["image"]
    tabular = sample["tabular"]
    target = sample["target"]

    print(f"    Sample Image Shape: {image.shape}")
    print(f"    Sample Tabular Shape: {tabular.shape}")
    print(f"    Sample Target: {target}")

    # Assertions
    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {image.shape}"
    assert (
        tabular.shape[0] == Config.N_TABULAR_FEATURES
    ), f"Expected {Config.N_TABULAR_FEATURES} tabular features, got {tabular.shape[0]}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = MCDSRNet().to(device)

    # Create a dummy batch
    batch_img = image.unsqueeze(0).to(device)  # [1, 3, H, W]
    batch_tab = tabular.unsqueeze(0).to(device)  # [1, 6]

    # Forward pass
    print("    Running forward pass...")
    mu, sigma = model(batch_img, batch_tab)

    print(f"    Model Output - Mu: {mu.item():.4f}, Sigma: {sigma.item():.4f}")

    # Assertions
    assert mu.shape == (1,), "Output Mu shape mismatch"
    assert sigma.shape == (1,), "Output Sigma shape mismatch"
    assert sigma.item() > 0, "Sigma (uncertainty) must be positive"

    # 4. Metric and Loss Verification
    print("\n[4] Verifying Loss and Metric...")

    # Test Loss
    target_gpu = target.to(device)  # [1]
    loss = metric_aligned_loss(mu, sigma, target_gpu)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Test Score Function (Competition Metric)
    # Simulate raw values (unscaled)
    y_true_raw = np.array([2500.0])
    y_pred_raw = np.array([2450.0])
    sigma_raw = np.array([150.0])

    score = score_function(y_true_raw, y_pred_raw, sigma_raw)
    print(f"    Calculated Score (Simulated): {score:.4f}")
    assert isinstance(score, float), "Score should be a float"

    # 5. Integration Test: Training Loop
    print("\n[5] Running Integration Test (Training Loop)...")

    # Get dataloaders (using debug=True for speed)
    train_loader, val_loader, test_loader, stats = get_dataloaders(debug=True)

    # Setup Optimizer
    backbone_params = [
        p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
    ]
    head_params = [
        p
        for n, p in model.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Run one epoch of training
    print("    Training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")

    # Run validation
    print("    Validating...")
    val_loss, val_metric = validate(model, val_loader, device, stats)
    print(f"    Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

    # 6. Inference Verification
    print("\n[6] Verifying Inference...")

    model.eval()
    results = []
    fvc_mean = stats["FVC_mean"]
    fvc_std = stats["FVC_std"]

    print("    Predicting on test set...")
    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            patient_weeks = batch["patient_week"]

            mu_scaled, sigma_scaled = model(img, tab)

            # Inverse Transform
            mu_raw = (mu_scaled * fvc_std + fvc_mean).cpu().numpy()
            sigma_raw = (sigma_scaled * fvc_std).cpu().numpy()

            # Clip sigma
            sigma_raw = np.maximum(sigma_raw, Config.MIN_UNCERTAINTY)

            for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_raw):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    assert len(results) > 0, "No predictions generated"
    print(f"    Generated {len(results)} predictions.")
    print(f"    First prediction: {results[0]}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
