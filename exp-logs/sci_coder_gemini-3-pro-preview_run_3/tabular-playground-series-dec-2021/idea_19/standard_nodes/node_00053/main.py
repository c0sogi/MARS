import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from library
from library.config import Config, set_seed
from library.data_utils import get_dataloaders, get_test_ids, feature_engineering
from library.train_utils import run_training, validate, generate_predictions


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_features = []

    # Collect data
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(y_batch.numpy())
            # Move features back to CPU for analysis
            all_features.append(X_batch.cpu().numpy())

    # Concatenate
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    X_val = np.concatenate(all_features)

    # Calculate Error (0 for correct, 1 for incorrect)
    errors = (y_pred != y_true).astype(int)

    print(f"Total Validation Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")

    # Retrieve feature names for meaningful reporting
    try:
        # Load a small sample of validation data to reconstruct feature names
        df_val_sample = pd.read_parquet(Config.VAL_PATH).head(5)
        df_val_sample = df_val_sample.drop(
            columns=["Id", "Cover_Type"], errors="ignore"
        )

        # Apply the same feature engineering
        df_processed = feature_engineering(df_val_sample)

        # Reconstruct the column order used in data_utils.py
        all_cols = df_processed.columns
        binary_cols = [
            c
            for c in all_cols
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        continuous_cols = [c for c in all_cols if c not in binary_cols]

        # The data loader stacks continuous then binary
        feature_names = continuous_cols + binary_cols

        if len(feature_names) != X_val.shape[1]:
            print(
                f"Warning: Feature name count ({len(feature_names)}) != X_val columns ({X_val.shape[1]}). Using indices."
            )
            feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    except Exception as e:
        print(f"Could not retrieve feature names: {e}. Using indices.")
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    # Compute correlation
    correlations = []
    for i in range(X_val.shape[1]):
        feat_col = X_val[:, i]

        # Skip constant columns to avoid NaN
        if np.std(feat_col) == 0:
            corr = 0.0
        else:
            # Point-biserial correlation
            corr = np.corrcoef(feat_col, errors)[0, 1]

        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features correlated with Error:")
    print(f"{'Feature':<40} {'Correlation':<10}")
    print("-" * 50)
    for name, corr in correlations[:10]:
        print(f"{name:<40} {corr:.6f}")

    return errors


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using full dataset to ensure we meet the high validation threshold.
    # The A100 GPU is sufficient to process this within the time limit.
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Determine input dimension
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]
    print(f"Input Dimension: {input_dim}")
    print(f"Classes: {Config.NUM_CLASSES}")

    # 3. Training
    print("Starting Training...")
    model = run_training(train_loader, val_loader, input_dim, Config.NUM_CLASSES)

    # 4. Final Validation
    print("Performing Final Validation...")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 6. Submission
    # Threshold defined in task description
    THRESHOLD = 0.9625041666666667

    if val_acc > THRESHOLD:
        print(
            f"\nValidation accuracy {val_acc} > {THRESHOLD}. Generating submission..."
        )
        test_ids = get_test_ids()
        generate_predictions(model, test_loader, test_ids, Config.SUBMISSION_PATH)
    else:
        print(f"\nValidation accuracy {val_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
