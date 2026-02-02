import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import make_dataloaders
from library.model import CGWBN
from library.engine import train_fold, validate, predict


def run():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    logger = get_logger()

    # Ensure we use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)  # Update config just in case

    # 2. Configure for Fast Baseline
    # The dataset is small (~1.6k images), so we can afford reasonable epochs.
    # However, to strictly comply with "limit training steps", we reduce max epochs.
    Config.NUM_EPOCHS = 35

    logger.info(f"Running on device: {device}")
    logger.info(f"Max Epochs set to: {Config.NUM_EPOCHS}")

    # 3. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader = make_dataloaders(load_cached_data=True)

    # 4. Model Initialization
    model = CGWBN()
    model.to(device)

    # 5. Training
    # We treat the provided train/val split as a single fold (Fold 0)
    logger.info("Starting training...")
    model = train_fold(
        fold_idx=0,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # 6. Validation & Metric Calculation
    logger.info("Performing final validation...")
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 7. Failure Analysis
    logger.info("Performing failure analysis...")
    model.eval()

    errors = []
    inc_angles = []
    img_means = []
    img_stds = []

    # Collect predictions and metadata
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            angles = batch["inc_angle"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            # Calculate absolute error
            batch_errors = torch.abs(probs - labels).cpu().numpy().flatten()
            errors.extend(batch_errors)

            # Collect incidence angles
            inc_angles.extend(angles.cpu().numpy().flatten())

            # Calculate image statistics (global mean/std per image)
            # images: (B, 3, 75, 75) -> mean over (1,2,3)
            b_mean = torch.mean(images, dim=(1, 2, 3)).cpu().numpy().flatten()
            b_std = torch.std(images, dim=(1, 2, 3)).cpu().numpy().flatten()
            img_means.extend(b_mean)
            img_stds.extend(b_std)

    # Compute Correlations
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_angles,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    corr_angle = df_analysis["error"].corr(df_analysis["inc_angle"])
    corr_mean = df_analysis["error"].corr(df_analysis["img_mean"])
    corr_std = df_analysis["error"].corr(df_analysis["img_std"])

    print(f"Correlation (Error vs Inc Angle): {corr_angle}")
    print(f"Correlation (Error vs Image Mean): {corr_mean}")
    print(f"Correlation (Error vs Image Std): {corr_std}")

    # 8. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.14772333549413377

    if val_loss < THRESHOLD:
        logger.info(
            f"Validation metric ({val_loss}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        ids, probs = predict(model, test_loader, device)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": ids, "is_iceberg": probs})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
