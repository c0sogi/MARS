import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR

# Import from provided library files
from library.config import Config
from library.utils import setup_logger, set_seed
from library.dataset import get_dataloaders
from library.model import AngleGatedResNet
from library.engine import Engine
from library.inference import generate_submission


def run():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    logger = setup_logger()
    set_seed(Config.SEED)

    logger.info("Initializing Fast Baseline Run...")

    # Override Config for speed
    Config.MAX_EPOCHS = 20  # Reduced from 100
    Config.PATIENCE = 5  # Reduced from 10
    Config.SWA_EPOCHS = 3  # Reduced from 12
    Config.N_FOLDS = 3  # Reduced from 5 for faster calibration

    # Threshold for submission
    TARGET_METRIC = 0.16918645240183008

    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Phase 1: Calibration (Find Optimal Epoch)
    # -------------------------------------------------------------------------
    logger.info("=== Step 1: Calibration (Finding Optimal Epoch) ===")
    # This uses CV to find the best epoch
    optimal_epoch = Engine.find_optimal_epoch()
    logger.info(f"Optimal Epoch determined: {optimal_epoch}")

    # -------------------------------------------------------------------------
    # 3. Verification Run (Train on Fixed Split -> Validate -> Failure Analysis)
    # -------------------------------------------------------------------------
    logger.info("=== Step 2: Verification & Failure Analysis ===")

    # We train a single model on the fixed train/val split to get the metric and analyze errors
    train_loader, val_loader = get_dataloaders(phase="calibration", load_cache=True)

    model = AngleGatedResNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # A. Standard Training
    logger.info(f"Training verification model for {optimal_epoch} epochs...")
    for epoch in range(optimal_epoch):
        train_loss, train_acc = Engine.train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{optimal_epoch} - Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
            )

    # B. SWA Training
    logger.info(f"Entering SWA phase for {Config.SWA_EPOCHS} epochs...")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    for i in range(Config.SWA_EPOCHS):
        Engine.train_one_epoch(model, train_loader, optimizer, criterion, device, i)
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Update BN
    Engine.custom_update_bn(train_loader, swa_model, device)

    # C. Validation & Metric Calculation
    # Note: Engine.validate returns loss (BCE) which is our metric
    val_loss, val_acc = Engine.validate(swa_model, val_loader, criterion, device)

    print(f"Final Validation Metric: {val_loss}")

    # D. Failure Analysis
    logger.info("Performing Failure Analysis...")
    swa_model.eval()

    errors = []
    angles = []
    b1_means = []
    b1_stds = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            ang = batch["inc_angle"].to(device)
            lbls = batch["label"].to(device).float().view(-1, 1)

            logits = swa_model(imgs, ang)
            probs = torch.sigmoid(logits)

            # Calculate absolute error
            batch_errors = torch.abs(probs - lbls).cpu().numpy().flatten()
            errors.extend(batch_errors)

            # Collect metadata/features for correlation
            angles.extend(ang.cpu().numpy().flatten())

            # Compute image stats on the fly (Band 1 is channel 0)
            # imgs shape: (B, 3, H, W)
            b1 = imgs[:, 0, :, :].cpu().numpy()
            b1_means.extend(np.mean(b1, axis=(1, 2)))
            b1_stds.extend(np.std(b1, axis=(1, 2)))

    # Convert to DF for correlation
    df_analysis = pd.DataFrame(
        {"error": errors, "inc_angle": angles, "b1_mean": b1_means, "b1_std": b1_stds}
    )

    correlations = df_analysis.corr()["error"].drop("error")
    logger.info("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 4. Phase 2: Production (Full Fit) & Submission
    # -------------------------------------------------------------------------
    if val_loss < TARGET_METRIC:
        logger.info(
            f"Validation Metric ({val_loss:.6f}) meets threshold ({TARGET_METRIC}). Proceeding to Submission."
        )

        # Train ensemble on full data
        # We use fewer models for the fast baseline (e.g., 2 instead of 5)
        Engine.train_full_fit_swa(optimal_epoch, num_models=2)

        # Generate Submission
        generate_submission(num_models=2)

    else:
        logger.warning(
            f"Validation Metric ({val_loss:.6f}) did NOT meet threshold ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    run()
