import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import UnifiedTransformer
from library.engine import train_model, generate_submission


def main():
    # 1. Setup
    print("Initializing pipeline...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=4
    )

    # 3. Model Initialization
    print("Initializing UnifiedTransformer...")
    model = UnifiedTransformer().to(device)

    # 4. Training
    # train_model handles the training loop, validation per epoch, and saving the best model.
    print("Starting training...")
    _ = train_model(model, train_loader, val_loader, device)

    # 5. Validation & Failure Analysis
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load the best model checkpoint for analysis
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    val_preds = []
    val_targets = []
    val_numericals = []

    # Inference loop for validation set
    with torch.no_grad():
        for batch in val_loader:
            numerical = batch["numerical"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(numerical, sequence)

            # Store results on CPU
            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_numericals.append(numerical.cpu().numpy())

    # Concatenate batches
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_numericals = np.concatenate(val_numericals, axis=0)

    # Compute Final Metric
    final_metric = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis: Correlation between Error and Features
    errors = np.abs(val_targets - val_preds)

    print("\nCorrelations between Error Magnitude and Numerical Features:")
    feature_names = Config.NUMERICAL_FEATURES
    correlations = []

    for i, feat_name in enumerate(feature_names):
        # Check for constant columns to avoid division by zero in correlation
        if np.std(val_numericals[:, i]) < 1e-9 or np.std(errors) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(val_numericals[:, i], errors)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.9920540777100928

    print("\n" + "=" * 40)
    print("SUBMISSION CHECK")
    print("=" * 40)

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric:.6f} > Threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Metric {final_metric:.6f} <= Threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
