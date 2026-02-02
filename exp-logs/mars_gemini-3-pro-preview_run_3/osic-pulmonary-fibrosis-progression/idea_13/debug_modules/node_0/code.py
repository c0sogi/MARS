import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, TargetScaler, MetricMonitor
from library.data import (
    TabularPreprocessor,
    add_baseline_info,
    OSICDataset,
    get_dataloaders,
    get_submission_loader,
)
from library.model import DSPRNet, laplace_nll_loss, train_one_epoch, validate


def run_demonstration():
    print("=== Starting DSPR-Net Pipeline Demonstration ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")
    seed_everything(42)

    # Override Config parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.MAX_TRAIN_SAMPLES = 12  # Use a tiny subset
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.DEBUG = True

    # Ensure directories exist
    Config.create_directories()
    print("    Configuration overrides applied (Batch Size: 4, Samples: 12).")

    # --------------------------------------------------------------------------
    # 2. Data Processing Components
    # --------------------------------------------------------------------------
    print("\n[2] Testing Data Processing Components...")

    # Load raw metadata manually to test preprocessors
    train_df = pd.read_csv(Config.TRAIN_CSV).head(Config.MAX_TRAIN_SAMPLES)
    train_df = add_baseline_info(train_df)
    print(f"    Loaded {len(train_df)} training samples.")

    # Test TabularPreprocessor
    print("    Testing TabularPreprocessor...")
    tab_prep = TabularPreprocessor()
    tab_prep.fit(train_df)
    feats = tab_prep.transform(train_df)

    # Validation
    assert feats.shape == (
        len(train_df),
        4,
    ), f"Expected tabular features shape ({len(train_df)}, 4), got {feats.shape}"
    assert feats.dtype == np.float32, "Tabular features must be float32."
    print("    -> TabularPreprocessor passed.")

    # Test TargetScaler
    print("    Testing TargetScaler...")
    scaler = TargetScaler()
    fvc_values = train_df["FVC"].values
    scaler.fit(fvc_values)
    transformed = scaler.transform(fvc_values)
    inversed = scaler.inverse_transform(transformed)

    # Validation
    assert np.allclose(
        fvc_values, inversed, atol=1e-4
    ), "TargetScaler inverse transform failed to recover original values."
    print("    -> TargetScaler passed.")

    # Test Dataset
    print("    Testing OSICDataset...")
    dataset = OSICDataset(
        train_df, tab_prep, scaler, mode="train", load_cached_data=True
    )

    sample = dataset[0]
    required_keys = ["image", "tabular", "t_rel", "patient_week", "target"]

    # Validation
    for k in required_keys:
        assert k in sample, f"Dataset sample missing key: {k}"

    assert sample["image"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {sample['image'].shape}"
    assert sample["tabular"].shape == (
        4,
    ), f"Tabular vector shape mismatch: {sample['tabular'].shape}"
    assert sample["target"].shape == (
        1,
    ), f"Target shape mismatch: {sample['target'].shape}"

    print("    -> OSICDataset passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture
    # --------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")
    device = Config.DEVICE
    model = DSPRNet().to(device)
    print(f"    Model initialized on {device}.")

    # Prepare dummy batch from the sample
    img = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
    tab = sample["tabular"].unsqueeze(0).to(device)  # (1, 4)
    t_rel = sample["t_rel"].unsqueeze(0).to(device)  # (1, 1)

    # Forward Pass
    print("    Running Forward Pass...")
    mu, sigma = model(img, tab, t_rel)

    # Validation
    assert mu.shape == (1,), f"Prediction (mu) shape mismatch: {mu.shape}"
    assert sigma.shape == (1,), f"Confidence (sigma) shape mismatch: {sigma.shape}"
    assert sigma.item() > 0, "Sigma (confidence) must be positive."
    print("    -> Forward Pass passed.")

    # Loss Calculation
    print("    Testing Laplace NLL Loss...")
    target = sample["target"].to(device).squeeze(-1)  # (1,)
    loss = laplace_nll_loss(target, mu, sigma)

    # Validation
    assert not torch.isnan(loss), "Loss returned NaN."
    assert loss.item() != 0, "Loss should not be exactly zero."
    print("    -> Loss function passed.")

    # --------------------------------------------------------------------------
    # 4. Training & Validation Loop
    # --------------------------------------------------------------------------
    print("\n[4] Testing Training & Validation Loop...")

    # Generate loaders using the library function (uses Config overrides)
    train_loader, val_loader, scaler_loaded, _ = get_dataloaders(load_cached_data=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train Step
    print("    Simulating one training epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"    -> Epoch complete. Train Loss: {train_loss:.4f}")

    # Validation Step
    print("    Simulating validation...")
    val_score = validate(model, val_loader, device, scaler_loaded)
    print(f"    -> Validation complete. Score: {val_score:.4f}")

    # --------------------------------------------------------------------------
    # 5. Metric Logic Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Metric Calculation Logic...")
    monitor = MetricMonitor()

    # Test Case: Perfect prediction with low confidence
    # True FVC = 2000, Pred FVC = 2000, Pred Sigma = 50
    # Sigma should be clipped to 70
    # Metric = - (sqrt(2)*0 / 70) - ln(sqrt(2)*70)
    #        = 0 - ln(98.9949) ≈ -4.595

    true_fvc = np.array([2000.0])
    pred_fvc = np.array([2000.0])
    pred_sigma = np.array([50.0])

    monitor.update(true_fvc, pred_fvc, pred_sigma)

    sigma_clipped = 70.0
    expected_metric = -np.log(np.sqrt(2) * sigma_clipped)

    assert np.isclose(
        monitor.avg, expected_metric, atol=1e-3
    ), f"Metric mismatch! Expected {expected_metric:.4f}, got {monitor.avg:.4f}"
    print("    -> MetricMonitor logic passed.")

    # --------------------------------------------------------------------------
    # 6. Submission Loader
    # --------------------------------------------------------------------------
    print("\n[6] Testing Submission Loader...")
    try:
        sub_loader = get_submission_loader(tab_prep, load_cached_data=True)
        batch = next(iter(sub_loader))

        # Validation
        assert "image" in batch
        assert "patient_week" in batch
        print("    -> Submission Loader passed.")
    except Exception as e:
        print(f"    -> Submission Loader check failed: {e}")
        raise e

    print("\n=== All demonstration steps completed successfully! ===")


if __name__ == "__main__":
    run_demonstration()
