import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_submission_loader
from library.model import MACOSR
from library.engine import run_training


def main():
    print("=== Starting MA-COSR Demo Script ===")

    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demo (Debug Mode)
    print("\n--- Step 1: Loading Data (Debug Mode) ---")
    # debug=True loads a small subset (50 rows) for speed
    train_loader, val_loader, scaler_stats = get_dataloaders(
        debug=True, load_cached_data=True
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    imgs = batch["image"].to(device)
    tabular = batch["tabular"].to(device)
    targets = batch["target"].to(device)

    print(f"Batch Image Shape: {imgs.shape}")  # Expected: (B, 3, 260, 260)
    print(f"Batch Tabular Shape: {tabular.shape}")  # Expected: (B, 5)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (B,)

    # Assertions for Data
    assert (
        imgs.dim() == 4 and imgs.shape[1] == 3
    ), "Image tensor has incorrect dimensions."
    assert (
        tabular.dim() == 2 and tabular.shape[1] == 5
    ), "Tabular tensor has incorrect dimensions."
    assert targets.dim() == 1, "Target tensor has incorrect dimensions."

    # 3. Model Instantiation & Forward Pass
    print("\n--- Step 2: Model Initialization & Forward Pass ---")
    model = MACOSR().to(device)

    # Run forward pass
    with torch.no_grad():
        pred_mu, pred_sigma = model(tabular, imgs)

    print(f"Prediction Mu Shape: {pred_mu.shape}")
    print(f"Prediction Sigma Shape: {pred_sigma.shape}")
    print(f"Sample Sigma Values: {pred_sigma[:5].cpu().numpy()}")

    # Assertions for Model
    assert pred_mu.shape == targets.shape, "Prediction shape mismatch."
    assert pred_sigma.shape == targets.shape, "Sigma shape mismatch."
    assert torch.all(pred_sigma > 0), "Sigma values must be positive."

    # 4. Training Loop Demo
    print("\n--- Step 3: Running Training Loop (2 Epochs) ---")

    # Setup Optimizer (Differential Learning Rates as per Config)
    optimizer = optim.AdamW(
        [
            {"params": model.clinical_anchor.parameters(), "lr": Config.LR_HEADS},
            {
                "params": model.visual_residual.backbone.parameters(),
                "lr": Config.LR_BACKBONE,
            },
            {
                "params": model.visual_residual.residual_head.parameters(),
                "lr": Config.LR_HEADS,
            },
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Setup Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Run Training (Short run for demo)
    best_score = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler_stats=scaler_stats,
        epochs=2,  # Limit to 2 epochs for speed
        patience=5,
    )

    print(f"Training completed. Best Validation Metric: {best_score:.4f}")

    # Verify Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"Checkpoint verified at: {Config.BEST_MODEL_PATH}")

    # 5. Inference & Submission Demo
    print("\n--- Step 4: Inference & Submission Generation ---")

    # Load Best Model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get Submission Loader
    sub_loader, sub_df = get_submission_loader(scaler_stats, load_cached_data=True)

    predictions = []
    confidences = []

    # Unnormalization stats
    fvc_mean = scaler_stats["fvc_mean"]
    fvc_std = scaler_stats["fvc_std"]

    with torch.no_grad():
        for batch in sub_loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)

            # Predict
            mu_norm, sigma_norm = model(tabular, imgs)

            # Unnormalize
            mu_ml = mu_norm * fvc_std + fvc_mean
            sigma_ml = sigma_norm * fvc_std

            predictions.extend(mu_ml.cpu().numpy())
            confidences.extend(sigma_ml.cpu().numpy())

    # Create Submission DataFrame
    # Note: sub_df from get_submission_loader is processed and might not match row order
    # if shuffle was True, but get_submission_loader sets shuffle=False.

    submission = pd.DataFrame(
        {
            "Patient_Week": sub_df["Patient_Week"],
            "FVC": predictions,
            "Confidence": confidences,
        }
    )

    # Save Submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(submission.head())

    # Assertions for Submission
    assert len(submission) == len(sub_df), "Submission row count mismatch."
    assert "Patient_Week" in submission.columns
    assert "FVC" in submission.columns
    assert "Confidence" in submission.columns

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
