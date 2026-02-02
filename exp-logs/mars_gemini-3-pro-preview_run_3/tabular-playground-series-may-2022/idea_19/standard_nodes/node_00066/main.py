import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import preprocess_data, ManufacturingDataset
from library.model import PIFEModel
from library.engine import train_model, evaluate, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, val_df, metadata):
    """
    Analyzes model failure modes by correlating prediction errors with input features.
    """
    print("\nPerforming Failure Analysis...")
    device = Config.DEVICE
    model.eval()

    val_probs = []
    val_targets = []

    # Generate predictions on validation set
    with torch.no_grad():
        for batch in val_loader:
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            logits_list = model(cont_x, cat_x)

            # Ensemble Averaging
            probs = torch.zeros_like(logits_list[0])
            for logits in logits_list:
                probs += torch.sigmoid(logits)
            probs /= len(logits_list)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(target.cpu().numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    correlations = {}

    # Correlate error with continuous features
    for col in metadata["cont_cols"]:
        if col in val_df.columns:
            feat_values = val_df[col].values
            # Ensure alignment
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Correlate error with categorical features
    for col in metadata["cat_cols"]:
        if col in val_df.columns:
            feat_values = val_df[col].values
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.6f}")


def main():
    # 1. Setup
    Config.set_seed()
    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # Load cached data if available to save time
    train_df, val_df, test_df, metadata = preprocess_data(load_cached_data=True)

    # 3. Dataset & DataLoader Creation
    train_ds = ManufacturingDataset(train_df, metadata, is_test=False)
    val_ds = ManufacturingDataset(val_df, metadata, is_test=False)
    test_ds = ManufacturingDataset(test_df, metadata, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = PIFEModel(metadata)

    # 5. Training
    # train_model handles the loop, optimizer, scheduler, and early stopping
    model = train_model(model, train_loader, val_loader)

    # 6. Final Validation Evaluation
    val_auc = evaluate(model, val_loader, Config.DEVICE)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, val_df, metadata)

    # 8. Submission
    # Strict threshold check as per requirements
    THRESHOLD = 0.9972038455079483

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader)
    else:
        print(
            f"\nValidation metric ({val_auc}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
