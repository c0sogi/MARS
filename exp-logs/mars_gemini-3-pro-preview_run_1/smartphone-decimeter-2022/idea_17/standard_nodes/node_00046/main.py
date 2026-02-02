import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Override Config before importing other modules to ensure settings propagate
from library.config import Config

Config.EPOCHS = 5  # Limit epochs for a fast baseline execution
Config.TRAIN_WINDOW_SIZE = 128
Config.TRAIN_WINDOW_STRIDE = 64

from library.utils import get_logger, fix_seed
from library.trainer import Trainer
from library.dataset import get_train_val_loaders

# Setup Logger
logger = get_logger("runfile")


def main():
    # Ensure reproducibility
    fix_seed(Config.SEED)

    logger.info("Starting Runfile execution...")

    # 1. Initialize Trainer and Train Model
    # The trainer handles model initialization, optimizer setup, and data loading internally for fit()
    trainer = Trainer(run_name="fast_baseline")

    logger.info("Starting Training...")
    # fit() returns the scaler used for normalization, which is needed for inference
    scaler = trainer.fit(debug=False)

    # 2. Validation Assessment
    logger.info("Starting Validation Assessment...")

    # Reload validation loader. load_cached_data=True ensures we use the preprocessed parquet file
    _, val_loader, _ = get_train_val_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False
    )

    model = trainer.model
    # Load the best checkpoint saved during training
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=trainer.device)
        )

    model.eval()
    device = trainer.device

    all_errors = []
    all_features = []

    # Feature names for analysis
    feature_names = Config.FEATURE_COLS

    # Inference loop for validation
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # Shape: (B, L, C)
            targets = batch["targets"].to(device)  # Shape: (B, L, 2) -> (dEast, dNorth)
            mask = batch["mask"].to(device)  # Shape: (B, L)

            # Forward pass
            # Model expects (B, C, L)
            features_t = features.transpose(1, 2)
            final_pred, _ = model(features_t)

            # Reshape predictions back to (B, L, 2)
            preds = final_pred.transpose(1, 2)

            # Calculate Euclidean distance error in meters
            # Error = sqrt((pred_east - target_east)^2 + (pred_north - target_north)^2)
            diff = preds - targets
            dist = torch.sqrt(torch.sum(diff**2, dim=2))  # (B, L)

            # Filter out padding using the mask
            valid_mask = mask.bool()

            # Flatten and move to CPU
            valid_dist = dist[valid_mask].cpu().numpy()
            valid_feats = features[valid_mask].cpu().numpy()

            all_errors.append(valid_dist)
            all_features.append(valid_feats)

    # Concatenate all batches
    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features)

    # Compute Final Metric
    p50 = np.percentile(all_errors, 50)
    p95 = np.percentile(all_errors, 95)
    final_metric = (p50 + p95) / 2

    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    logger.info("Performing Failure Analysis...")
    correlations = {}

    # Calculate correlation between each feature and the error magnitude
    for i, feat_name in enumerate(feature_names):
        feat_values = all_features[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_values, all_errors)
        correlations[feat_name] = corr

    # Sort by absolute correlation strength
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features correlated with Error Magnitude:")
    for name, corr in sorted_corrs[:5]:
        print(f"  {name}: {corr:.4f}")

    # 4. Submission Generation
    THRESHOLD = 3.802240262877392

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        # trainer.predict handles test data loading, inference, coordinate reconstruction, and saving CSV
        trainer.predict(scaler)
    else:
        logger.warning(
            f"Validation metric {final_metric} is >= threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
