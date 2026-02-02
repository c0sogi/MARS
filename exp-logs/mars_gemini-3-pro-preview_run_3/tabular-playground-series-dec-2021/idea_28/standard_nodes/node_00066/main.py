import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.data_utils import process_data, feature_engineering, CoverTypeDataset
from library.model_utils import set_seed
from library.train_utils import run_training, evaluate, predict_and_submit


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    # We use the full dataset to ensure we can hit the high accuracy threshold.
    # We limit epochs to 25 to ensure the run completes quickly (Fast Baseline).
    print("Starting Training Pipeline...")
    model = run_training(
        epochs=25,
        max_train_samples=None,
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
    )

    # 3. Validation & Metric Calculation
    print("\nPerforming Final Validation...")
    # Load data (should be cached now)
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=True
    )

    # Create Validation DataLoader
    val_dataset = CoverTypeDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Final Validation Metric: {val_acc}")

    # 4. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    # Calculate errors (1 for incorrect, 0 for correct)
    errors = (y_pred != y_val).astype(int)
    total_errors = errors.sum()
    print(
        f"Total Misclassified Samples: {total_errors} out of {len(errors)} ({total_errors/len(errors):.4%})"
    )

    if total_errors > 0 and total_errors < len(errors):
        # Reconstruct feature names to make the report useful
        # We need to temporarily load a small dataframe to get columns after engineering
        try:
            df_tmp = pd.read_parquet(Config.TRAIN_PATH)
            df_tmp = feature_engineering(df_tmp)
            exclude_cols = set(Config.BINARY_COLS + [Config.ID_COL, Config.TARGET_COL])
            continuous_cols = [c for c in df_tmp.columns if c not in exclude_cols]
            # The order in data_utils is hstack([continuous, binary])
            feature_names = continuous_cols + Config.BINARY_COLS

            # Verify shape
            if len(feature_names) != X_val.shape[1]:
                print(
                    f"Warning: Feature name count ({len(feature_names)}) does not match X_val shape ({X_val.shape[1]}). Using indices."
                )
                feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
        except Exception as e:
            print(f"Could not reconstruct feature names: {e}. Using indices.")
            feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

        # Calculate correlations
        print("Calculating correlations between features and error magnitude...")
        correlations = []
        for i in range(X_val.shape[1]):
            feat_col = X_val[:, i]
            # Point-biserial correlation (Pearson between continuous and binary)
            # Handle constant features to avoid NaN
            if np.std(feat_col) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feat_col, errors)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            correlations.append((feature_names[i], corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(
            "\nTop 10 Features Associated with Model Failure (Correlation with Error):"
        )
        print(f"{'Feature':<40} {'Correlation':<15}")
        print("-" * 55)
        for name, corr in correlations[:10]:
            print(f"{name[:38]:<40} {corr:.6f}")
    else:
        print("Skipping correlation analysis (0 errors or 100% errors).")

    # 5. Submission Logic
    THRESHOLD = 0.9625222222222222
    submission_path = Config.SUBMISSION_PATH

    if val_acc > THRESHOLD:
        print(f"\nValidation metric {val_acc} exceeds threshold {THRESHOLD}.")
        # run_training already generated the submission file.
        # We verify it exists.
        if os.path.exists(submission_path):
            print(f"Submission file confirmed at {submission_path}.")
        else:
            print("Submission file missing. Regenerating...")
            _, _, test_loader, _, test_ids = get_dataloaders(load_cached_data=True)
            predict_and_submit(model, test_loader, test_ids, device)
    else:
        print(f"\nValidation metric {val_acc} does not exceed threshold {THRESHOLD}.")
        if os.path.exists(submission_path):
            print("Removing submission file...")
            os.remove(submission_path)
        print("Submission discarded.")


if __name__ == "__main__":
    main()
