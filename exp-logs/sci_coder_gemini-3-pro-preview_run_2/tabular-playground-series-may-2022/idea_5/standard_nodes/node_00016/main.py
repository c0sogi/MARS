import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Ensure library is importable
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import GatedMLP
from library.trainer import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for fast baseline execution while maintaining performance
    Config.EPOCHS = 50
    SUBMISSION_FILE = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Running with Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nLoading Data...")
    # Use cached data as requested to save time
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model & Training
    # --------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = GatedMLP()

    print("\nStarting Training...")
    trainer = Trainer(model)

    # Explicitly pass modified epochs to override default in function signature
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nRunning Final Validation & Failure Analysis...")

    # Set model to eval mode
    model.eval()
    device = torch.device(Config.DEVICE)

    val_preds = []
    val_targets = []
    val_cont_features = []

    # Inference on validation set to gather data for analysis
    with torch.no_grad():
        for batch in val_loader:
            x_cat = batch["cat"].to(device)
            x_cont = batch["cont"].to(device)
            y = batch["target"].to(device)

            preds = model(x_cat, x_cont)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_cont_features.append(x_cont.cpu().numpy())

    # Concatenate results
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_cont_features = np.concatenate(val_cont_features, axis=0)

    # Calculate Metric
    final_auc = roc_auc_score(val_targets, val_preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Continuous Features
    errors = np.abs(val_targets - val_preds)
    cont_feature_names = Config.CONT_FEATURES

    print("\nFailure Analysis (Error Correlation with Continuous Features):")
    correlations = []

    for i, name in enumerate(cont_feature_names):
        feat_values = val_cont_features[:, i]
        # Check for zero variance to avoid warnings
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            # Use numpy for correlation
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:5]:
        print(f"Feature: {name}, Correlation with Error: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9952920431395679

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        test_preds = trainer.predict(test_loader)

        # Load IDs using metadata
        test_meta = pd.read_csv(Config.TEST_METADATA)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "target": test_preds})

        # Save
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")

    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
