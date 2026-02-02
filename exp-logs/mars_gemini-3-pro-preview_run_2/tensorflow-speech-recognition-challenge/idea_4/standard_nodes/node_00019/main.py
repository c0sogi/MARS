import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.trainer import Trainer
from library.dataset import get_dataloaders


def main():
    # 1. Setup
    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on device: {device}")

    # 2. Train
    # Instantiate the Trainer and fit the model.
    # The Config defines 15 epochs which is efficient for this dataset size on an A100.
    # We use the full dataset to ensure we meet the high accuracy threshold.
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Failure Analysis
    print("\nRunning validation and failure analysis...")

    # Ensure the best model is loaded for analysis
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model checkpoint not found.")
        return

    # Load checkpoint
    load_checkpoint(Config.BEST_MODEL_PATH, trainer.model, device=device)
    trainer.model.eval()

    # Get validation loader (use cached data for speed)
    _, val_loader, _ = get_dataloaders(debug=False, load_cached=True)

    all_preds = []
    all_labels = []
    all_probs = []

    # Store simple features for failure analysis
    # We will correlate error magnitude with Spectrogram Mean and Std
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = trainer.model(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Extract probability of the true class
            # labels: (B,), probs: (B, C) -> gather along dim 1
            true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            all_probs.extend(true_probs.cpu().numpy())

            # Calculate input feature statistics (Spectrogram properties)
            # features: (B, 1, F, T) -> Flatten spatial dims to calculate stats per sample
            flat_feats = features.view(features.size(0), -1)
            feat_means.extend(flat_feats.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_feats.std(dim=1).cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    feat_means = np.array(feat_means)
    feat_stds = np.array(feat_stds)

    # Calculate Final Validation Metric
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {final_acc}")

    # Perform Failure Analysis
    # Error Magnitude = 1.0 - Probability assigned to the correct class
    error_magnitude = 1.0 - all_probs

    # Calculate correlations using numpy
    # np.corrcoef returns a matrix [[1, corr], [corr, 1]]
    corr_mean = np.corrcoef(error_magnitude, feat_means)[0, 1]
    corr_std = np.corrcoef(error_magnitude, feat_stds)[0, 1]

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Spectrogram Mean Intensity: {corr_mean}")
    print(f"  Spectrogram Contrast (Std): {corr_std}")

    # 4. Submission
    # Check against the required threshold
    THRESHOLD = 0.9853666694539677

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric {final_acc} > {THRESHOLD}. Generating submission..."
        )
        # The trainer.predict method handles test loading and submission file generation
        trainer.predict()
    else:
        print(f"\nValidation metric {final_acc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
