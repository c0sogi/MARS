import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MCDSRNet, train_one_epoch, validate


def run_training(debug=Config.DEBUG, patience=10):
    """
    Main execution function for training and inference.

    Args:
        debug (bool): If True, runs on a subset of data.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading
    print(f"Loading data (Debug={debug})...")
    train_loader, val_loader, test_loader, stats = get_dataloaders(debug=debug)

    # 3. Model Setup
    device = torch.device(Config.DEVICE)
    model = MCDSRNet().to(device)

    # 4. Optimizer & Scheduler Setup
    # Differential Learning Rates: Lower for backbone, higher for custom heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validation Step
        val_loss, val_metric = validate(model, val_loader, device, stats)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Metric: {best_metric:.10f}")

    # 6. Inference / Prediction
    print("Generating predictions on test set...")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    results = []
    fvc_mean = stats["FVC_mean"]
    fvc_std = stats["FVC_std"]

    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            patient_weeks = batch["patient_week"]

            # Predict
            mu_scaled, sigma_scaled = model(img, tab)

            # Inverse Transform
            mu_raw = (mu_scaled * fvc_std + fvc_mean).cpu().numpy()
            sigma_raw = (sigma_scaled * fvc_std).cpu().numpy()

            # Post-processing: Clip sigma strictly for submission
            # Note: The metric clips at 70, so we ensure our output respects this lower bound
            sigma_raw = np.maximum(sigma_raw, Config.MIN_UNCERTAINTY)

            for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_raw):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # 7. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())
