import sys
import os
import numpy as np
import pandas as pd
import torch
import random
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import train_model, predict

# -------------------------------------------------------------------------
# Orchestration Script
# -------------------------------------------------------------------------


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Configuration & Setup
    set_seed(Config.SEED)

    # Full Training Configuration (Cite 00021)
    # We use the full dataset to maximize performance, as data scale > model complexity.
    Config.EPOCHS = 30
    Config.BATCH_SIZE = 4096

    # Use full dataset (None means no truncation)
    MAX_TRAIN_SAMPLES = None

    print(f"Starting Full Training Run (Epochs={Config.EPOCHS}, Max Samples=All)...")

    # 2. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Implement Subsampling for Fast Baseline
    if MAX_TRAIN_SAMPLES is not None and len(train_loader.dataset) > MAX_TRAIN_SAMPLES:
        print(f"Subsampling training set to {MAX_TRAIN_SAMPLES} samples...")
        full_X, full_y = train_loader.dataset.tensors

        # Shuffle indices to ensure random subset
        indices = torch.randperm(len(full_X))[:MAX_TRAIN_SAMPLES]

        small_X = full_X[indices]
        small_y = full_y[indices]

        # Recreate DataLoader
        train_dataset = torch.utils.data.TensorDataset(small_X, small_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

    # 3. Model Training
    # train_model handles optimization, scheduling, and early stopping
    model = train_model(train_loader, val_loader, input_dim)

    # 4. Validation & Metric Calculation
    print("Computing Final Validation Metric...")
    device = torch.device(Config.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    # Inference loop (no_grad for speed/memory)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Accuracy
    accuracy = (all_preds == all_labels).mean()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load validation features for correlation analysis
    # We load directly from cache to get the full numpy array aligned with val_loader
    if os.path.exists(Config.CACHE_VAL_X):
        val_X = np.load(Config.CACHE_VAL_X)
    else:
        # Fallback if cache missing (unlikely given get_dataloaders ran)
        print("Validation cache not found. Skipping detailed failure analysis.")
        val_X = None

    if val_X is not None:
        # Create Error Mask (1 = Error, 0 = Correct)
        error_mask = (all_preds != all_labels).astype(int)

        # Define Feature Names
        # Order: Continuous + New_Features + Binary
        feature_names = (
            Config.CONTINUOUS_COLS + Config.NEW_FEATURES + Config.BINARY_COLS
        )

        correlations = []

        # Calculate Point-Biserial Correlation for each feature
        # Iterate only up to the number of columns present
        num_cols = min(val_X.shape[1], len(feature_names))

        for i in range(num_cols):
            feat_col = val_X[:, i]
            # Handle potential constant columns to avoid NaN
            if np.std(feat_col) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feat_col, error_mask)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

            correlations.append((feature_names[i], corr))

        # Sort by magnitude of correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 10 Features Correlated with Prediction Errors:")
        print(f"{'Feature':<40} {'Correlation':<10}")
        print("-" * 50)
        for name, corr in correlations[:10]:
            print(f"{name:<40} {corr:.6f}")

    # 6. Conditional Submission
    # Threshold defined in task
    THRESHOLD = 0.9622416666666667

    if accuracy > THRESHOLD:
        print(f"\nValidation metric ({accuracy}) exceeds threshold ({THRESHOLD}).")
        predict(model, test_loader, test_ids)
    else:
        print(
            f"\nValidation metric ({accuracy}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
