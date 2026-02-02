import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import warnings
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger, kl_divergence_score
from library.data_loader import get_dataloaders
from library.model import DualScaleSpectrogramNet
from library.engine import train_one_epoch, validate, generate_submission

# Suppress warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("train_logger").setLevel(logging.ERROR)
logging.getLogger("data_loader").setLevel(logging.ERROR)


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline
    Config.EPOCHS = 6
    Config.TRAIN_SUBSET_SIZE = 10000  # Use subset for speed
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4

    # Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Logger (Minimal output)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "run.log"))
    logger.info("Starting Fast Baseline Run...")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Subsample training data for fast baseline
    if len(train_df) > Config.TRAIN_SUBSET_SIZE:
        train_df = train_df.sample(
            n=Config.TRAIN_SUBSET_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        logger.info(f"Subsampled training data to {len(train_df)} rows.")

    # Get DataLoaders
    # This handles caching automatically in ./working
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # ==========================================
    # 3. Model Training
    # ==========================================
    device = Config.DEVICE
    model = DualScaleSpectrogramNet(Config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_kl = float("inf")

    logger.info(f"Training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_kl = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config
        )

        # Validate
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"Epoch {epoch+1}: Train KL={train_kl:.5f}, Val KL={val_kl:.5f}")

        # Save Best Model
        if val_kl < best_kl:
            best_kl = val_kl
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ==========================================
    # 4. Final Validation & Metric
    # ==========================================
    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()

    # Collect predictions and targets for analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            x_eeg = inputs[0].to(device)
            x_spec = inputs[1].to(device)
            target_batch = targets.to(device)

            outputs = model((x_eeg, x_spec))

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    # We use the provided utility which clips and handles log(0)
    final_metric = kl_divergence_score(all_targets, all_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")

    # Calculate KL per sample manually to correlate
    # KL(P || Q) = sum(P * log(P/Q))
    epsilon = 1e-15
    y_pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    kl_per_sample = []
    for i in range(len(all_targets)):
        p = all_targets[i]
        q = y_pred_clipped[i]
        # Mask for p > 0
        mask = p > 0
        if mask.any():
            sample_kl = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
        else:
            sample_kl = 0.0
        kl_per_sample.append(sample_kl)

    kl_per_sample = np.array(kl_per_sample)

    # Correlate with Metadata features
    # Ensure val_df aligns with loader (loader is sequential for val)
    if len(val_df) == len(kl_per_sample):
        features_to_check = ["total_votes", "eeg_label_offset_seconds"]
        print("-" * 30)
        print("Failure Analysis (Correlation with Error):")
        for feat in features_to_check:
            if feat in val_df.columns:
                # Handle NaNs if any
                feat_vals = val_df[feat].fillna(0).values
                corr, _ = pearsonr(feat_vals, kl_per_sample)
                print(f"  Correlation with {feat}: {corr:.4f}")
        print("-" * 30)
    else:
        logger.warning(
            "Validation dataframe length mismatch. Skipping detailed correlation."
        )

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 1.0065

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader, test_df, Config, logger)
    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
