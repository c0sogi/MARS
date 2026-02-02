import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer, set_seed
from library.dataset import get_datasets


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    set_seed()

    # Configure for a fast but high-performance baseline
    # 30 epochs is sufficient for convergence on this dataset with ResMLP
    # and fits comfortably within the runtime limits on an A100.
    Config.EPOCHS = 30
    Config.DEBUG = False

    print(f"Starting execution with Device: {Config.DEVICE}")
    print(f"Training for {Config.EPOCHS} epochs with batch size {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    trainer = Trainer()

    # Train the model
    # Using cached data to save preprocessing time
    trainer.fit(epochs=Config.EPOCHS, patience=5, load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load validation data manually to access features and targets for analysis
    _, val_ds, _ = get_datasets(load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensure we are using the best saved model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        )

    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_continuous = []

    # Inference loop without gradient calculation for speed
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(trainer.device)
            categorical = batch["categorical"].to(trainer.device)
            targets = batch["target"].to(trainer.device)

            outputs = trainer.model(continuous, categorical)
            probs = torch.sigmoid(outputs).squeeze(1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # Store continuous features for correlation analysis
            all_continuous.append(continuous.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_continuous = np.concatenate(all_continuous)

    # Calculate and print the mandatory validation metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlate error magnitude with input features
    errors = np.abs(all_targets - all_preds)

    print("\nFailure Analysis (Top 5 Feature Correlations with Error):")
    feature_correlations = []

    # Calculate correlation for each continuous feature
    # Note: all_continuous corresponds to sorted f_00..f_30 (excluding f_27)
    for i in range(all_continuous.shape[1]):
        feat_values = all_continuous[:, i]
        if np.std(feat_values) > 0:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        else:
            corr = 0.0
        feature_correlations.append((i, corr))

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for idx, corr in feature_correlations[:5]:
        print(f"Feature Index {idx}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9948596381822921

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
