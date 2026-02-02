import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data import (
    CTScanProcessor,
    TabularScaler,
    PulmonaryDataset,
    get_baseline_lookup,
    prepare_inference_dataframe,
)
from library.model import CAPNet
from library.train import train_one_epoch, evaluate, laplace_log_likelihood_loss
from library.utils import metric_laplace_log_likelihood


def main():
    print("=== Starting Pulmonary Fibrosis Model Demonstration ===\n")

    # 1. Setup Configuration
    print("[1] Setting up Configuration...")
    Config.setup()

    # Override configuration for rapid demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    print(f"    Device: {Config.DEVICE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Loading and Processing
    print("\n[2] Initializing Data Processing Components...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"    Loaded {len(train_df)} training rows.")

    # Initialize Processor
    processor = CTScanProcessor(img_size=Config.IMG_SIZE, num_slices=Config.NUM_SLICES)

    # Initialize and Fit Scaler
    scaler = TabularScaler()
    scaler.fit(train_df)
    print("    TabularScaler fitted.")

    # Verify Scaler Logic
    transformed_data = scaler.transform(train_df)
    weeks_mean = transformed_data["Weeks"].mean()
    weeks_std = transformed_data["Weeks"].std()
    print(f"    Scaled 'Weeks' - Mean: {weeks_mean:.4f}, Std: {weeks_std:.4f}")

    # Assert scaling is working (Mean should be close to 0, Std close to 1)
    # Note: With very small sample sizes (e.g. 20), std might not be exactly 1 if using ddof=1 vs 0,
    # but mean should be effectively 0.
    assert np.abs(weeks_mean) < 1e-4, "Scaler failed: Mean is not approx 0"

    # Create Baseline Lookup
    baseline_lookup = get_baseline_lookup(train_df)

    # Instantiate Dataset
    train_dataset = PulmonaryDataset(
        train_df, processor, scaler, baseline_lookup, mode="train"
    )

    # Verify Dataset __getitem__
    print("    Verifying Dataset sample output...")
    sample = train_dataset[0]

    # Check shapes
    # Image: (3, H, W)
    assert sample["image"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {sample['image'].shape}"
    # Meta Categorical: (2,) [Sex, Smoking]
    assert sample["meta_cat"].shape == (2,), "Incorrect meta_cat shape"
    # Meta Numerical: (2,) [Age, Percent]
    assert sample["meta_num"].shape == (2,), "Incorrect meta_num shape"

    print("    Dataset verification successful.")

    # 3. Model Initialization
    print("\n[3] Initializing CAPNet Model...")
    model = CAPNet().to(Config.DEVICE)
    print("    Model initialized and moved to device.")

    # 4. Forward Pass Simulation
    print("\n[4] Running Forward Pass Simulation...")
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Get a batch
    batch = next(iter(train_loader))

    # Move to device
    image = batch["image"].to(Config.DEVICE)
    meta_cat = batch["meta_cat"].to(Config.DEVICE)
    meta_num = batch["meta_num"].to(Config.DEVICE)
    baseline_fvc = batch["baseline_fvc_scaled"].to(Config.DEVICE)
    weeks = batch["weeks_scaled"].to(Config.DEVICE)
    target = batch["target_fvc_scaled"].to(Config.DEVICE)

    # Forward
    mu, sigma = model(image, meta_cat, meta_num, baseline_fvc, weeks)

    print(f"    Input Batch Size: {image.size(0)}")
    print(f"    Output mu shape: {mu.shape}")
    print(f"    Output sigma shape: {sigma.shape}")

    assert mu.shape == (image.size(0),), "Output mu shape mismatch"
    assert sigma.shape == (image.size(0),), "Output sigma shape mismatch"

    # 5. Loss and Metric Calculation
    print("\n[5] Verifying Loss and Metric functions...")

    # Loss
    loss = laplace_log_likelihood_loss(mu, sigma, target)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"

    # Metric (requires numpy arrays)
    y_true = target.detach().cpu().numpy()
    y_pred = mu.detach().cpu().numpy()
    y_sigma = sigma.detach().cpu().numpy()

    # Note: We are passing scaled values here just to test the math,
    # in real eval we unscale first.
    metric_val = metric_laplace_log_likelihood(y_true, y_pred, y_sigma)
    print(f"    Calculated Metric (Scaled inputs): {metric_val:.4f}")
    assert np.isfinite(metric_val), "Metric is not finite"

    # 6. Training Loop Execution
    print("\n[6] Executing Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    avg_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
    print(f"    Epoch 1 Completed. Average Loss: {avg_loss:.4f}")

    # 7. Evaluation Execution
    print("\n[7] Executing Evaluation Loop...")
    # Use the same loader as val for demo purposes
    eval_score = evaluate(model, train_loader, scaler, Config.DEVICE)
    print(f"    Evaluation Score (Unscaled inputs): {eval_score:.4f}")

    # 8. Inference Preparation
    print("\n[8] Verifying Inference Preparation...")
    if os.path.exists(Config.SAMPLE_SUBMISSION) and os.path.exists(Config.TEST_CSV):
        inf_df = prepare_inference_dataframe(Config.SAMPLE_SUBMISSION, Config.TEST_CSV)
        print(f"    Inference DataFrame created with {len(inf_df)} rows.")
        print(f"    Columns: {list(inf_df.columns)}")

        # Check for required columns for Dataset
        assert "Patient" in inf_df.columns
        assert "Weeks" in inf_df.columns
        assert "Base_FVC" in inf_df.columns
    else:
        print("    Skipping inference check (missing input files).")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
