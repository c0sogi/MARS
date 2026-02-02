import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.architecture import DEVICE, SEED

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    # --------------------------------------------------------------------------
    # 1. Training
    # --------------------------------------------------------------------------
    # We use the full 40 epochs recommended in the Idea.
    # On an A100, tabular training of this size is very fast (minutes).
    # This ensures we have the best chance of beating the high threshold.
    trainer = Trainer()
    trainer.fit(epochs=40, batch_size=1024, patience=10)

    # --------------------------------------------------------------------------
    # 2. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    # Load validation data
    _, val_loader, _, _ = get_dataloaders(batch_size=1024, load_cached_data=True)

    trainer.model.eval()
    val_preds = []
    val_targets = []
    val_features_list = []

    # Run inference on validation set
    # We also collect features for failure analysis
    with torch.no_grad():
        for batch in val_loader:
            num_x = batch["numerical"].to(DEVICE)
            cat_x = batch["categorical"].to(DEVICE)
            target = batch["target"]

            output = trainer.model(num_x, cat_x)

            val_preds.append(output.cpu().numpy())
            val_targets.append(target.numpy())
            val_features_list.append(num_x.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_features = np.concatenate(val_features_list)

    # Calculate and print metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 3. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nFailure Analysis (Correlation with Error):")

    # Calculate Error Magnitude
    # Target is binary (0 or 1), preds are probabilities.
    # Error is absolute difference.
    errors = np.abs(val_targets.flatten() - val_preds.flatten())

    # Reconstruct feature names (f_00 to f_30, excluding f_27)
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    # Calculate correlation for each numerical feature
    for idx, feat_name in enumerate(feature_names):
        feat_values = val_features[:, idx]
        # Handle potential constant features to avoid NaN correlation
        if np.std(feat_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9971578675877165

    if final_auc > THRESHOLD:
        trainer.predict(batch_size=1024)
    else:
        print(
            f"Validation AUC {final_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
