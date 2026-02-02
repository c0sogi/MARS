import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import OSPRNet
from library.train import (
    train_epoch,
    validate,
    generate_submission,
    metric_aligned_laplace_loss,
)


def run_demo():
    print("=== Starting OSPR-Net Demonstration ===\n")

    # 1. Configuration Override for Speed
    # We use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute load for demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Initialize directories
    Config.setup()
    seed_everything(Config.SEED)
    print(
        f"Configuration: Device={Config.DEVICE}, Batch Size={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}"
    )

    # 2. Metric Verification
    print("\n--- Verifying Metric Logic ---")
    # Case: Perfect prediction, high confidence
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])

    # Formula: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    # delta = 0
    # sigma_clipped = 100
    # expected = -0 - ln(141.42...) = -4.9517...
    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    print(f"Metric Check (Perfect Pred): {score:.4f}")

    expected_score = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction."

    # Case: Large Error, clipped at 1000
    y_true_bad = np.array([2000.0])
    y_pred_bad = np.array([4000.0])  # Delta = 2000 -> Clipped to 1000
    sigma_bad = np.array([70.0])

    score_bad = laplace_log_likelihood_metric(y_true_bad, y_pred_bad, sigma_bad)
    print(f"Metric Check (Clipped Error): {score_bad:.4f}")

    # delta = 1000, sigma = 70
    # term1 = sqrt(2) * 1000 / 70 = 20.203
    # term2 = ln(sqrt(2) * 70) = 4.595
    # total = -24.798
    expected_bad = -(np.sqrt(2) * 1000 / 70) - np.log(np.sqrt(2) * 70)
    assert np.isclose(
        score_bad, expected_bad, atol=1e-4
    ), "Metric calculation mismatch for clipped error."
    print("Metric logic verified.")

    # 3. Data Pipeline
    print("\n--- Initializing Data Loaders ---")
    # We disable loading cached data to demonstrate processing (or rely on it if files exist)
    # For speed in this demo, we'll let it process a few items.
    train_loader, val_loader, sub_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Submission batches: {len(sub_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    imgs, tabs, targets = batch

    print(
        f"Batch Shapes -> Image: {imgs.shape}, Tabular: {tabs.shape}, Target: {targets.shape}"
    )

    # Assertions
    assert imgs.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert tabs.shape == (
        Config.BATCH_SIZE,
        5,
    )  # 5 features: BaseFVC, Time, Age, Sex, Smoke
    assert targets.shape == (Config.BATCH_SIZE,)
    print("Data shapes verified.")

    # 4. Model Initialization & Logic
    print("\n--- Initializing Model ---")
    model = OSPRNet().to(Config.DEVICE)

    # Verify Zero Initialization of Visual Residual
    # The visual stream should output 0 correction initially
    print("Verifying Visual Residual Zero-Initialization...")
    imgs_dev = imgs.to(Config.DEVICE)
    tabs_dev = tabs.to(Config.DEVICE)

    with torch.no_grad():
        # Get full output
        mu, sigma = model(imgs_dev, tabs_dev)

        # Get specific stream outputs
        clin_out = model.clinical_anchor(tabs_dev)
        res_out = model.visual_residual(imgs_dev, tabs_dev)

        # res_out should be all zeros (or extremely close due to float precision)
        # res_out shape is (B, 2) -> [delta_mu, delta_sigma]
        print(f"Residual Mean Abs Value: {res_out.abs().mean().item():.6f}")
        assert torch.allclose(
            res_out, torch.zeros_like(res_out), atol=1e-5
        ), "Visual Residual head not initialized to zero!"

        # Consequently, the final mu should equal the clinical mu
        assert torch.allclose(
            mu, clin_out[:, 0], atol=1e-5
        ), "Model output does not match clinical anchor at initialization."

    print("Model logic verified.")

    # 5. Training Loop Demonstration
    print("\n--- Running Training Epoch ---")
    # Setup Optimizer
    backbone_params_ids = list(map(id, model.visual_residual.backbone.parameters()))
    backbone_params = [
        p
        for p in model.parameters()
        if id(p) in backbone_params_ids and p.requires_grad
    ]
    head_params = [
        p
        for p in model.parameters()
        if id(p) not in backbone_params_ids and p.requires_grad
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Run 1 Epoch
    train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Run Validation
    print("\n--- Running Validation ---")
    val_score = validate(model, val_loader, Config.DEVICE)
    print(f"Validation Score: {val_score:.4f}")

    # Assert score is reasonable (negative value)
    assert val_score < 0, "Validation score should be negative (log likelihood)."

    # 6. Inference & Submission
    print("\n--- Generating Submission ---")
    generate_submission(model, sub_loader, Config.DEVICE)

    # Verify File
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file loaded. Shape: {df_sub.shape}")
        print(df_sub.head())

        # Checks
        assert "Patient_Week" in df_sub.columns
        assert "FVC" in df_sub.columns
        assert "Confidence" in df_sub.columns
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"
        assert (df_sub["Confidence"] >= 70).all(), "Confidence values below 70 found"

        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
