import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import copy

from library.config import Config, IcebergDataset
from library.model import MDSWBN
from library.data_loader import get_data, get_test_loader
from library.utils import seed_everything
from library.train import run_fold, predict_fold


def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Configuration
    EPOCHS = Config.EPOCHS
    BATCH_SIZE = Config.BATCH_SIZE
    DEVICE = Config.DEVICE

    print(f"Running on device: {DEVICE}")

    # 1. Load Metadata to define splits
    print("Loading metadata...")
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata files not found.")

    df_train_meta = pd.read_csv(train_meta_path)
    df_val_meta = pd.read_csv(val_meta_path)

    n_train = len(df_train_meta)
    n_val = len(df_val_meta)
    print(f"Metadata counts - Train: {n_train}, Val: {n_val}")

    # 2. Load Data
    # get_data returns the combined training data (Train + Val) as processed by load_and_process_data
    print("Loading and processing data...")
    X_all, y_all, inc_all, X_test, inc_test, test_ids = get_data(debug=False)

    # 3. Split Data
    # load_and_process_data concatenates train then val, so we slice accordingly
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    inc_train = inc_all[:n_train]

    X_val = X_all[n_train:]
    y_val = y_all[n_train:]
    inc_val = inc_all[n_train:]

    # 4. K-Fold Training (Cite Lesson 00052)
    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation on Training Set...")
    models = []

    for fold in range(Config.N_FOLDS):
        # run_fold handles the internal splitting of X_train into train/val for early stopping
        print(f"\n--- Fold {fold + 1} ---")
        wts = run_fold(fold, X_train, y_train, inc_train, epochs=EPOCHS, debug=False)

        # Load model for inference
        model = MDSWBN().to(DEVICE)
        model.load_state_dict(wts)
        model.eval()
        models.append(model)

    print("\nTraining complete. Ensembling on Hold-out Validation Set...")

    # 5. Ensemble Evaluation on Hold-out Set (Cite Lesson 00036)
    # Create loader for hold-out validation set
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_preds_accumulator = np.zeros((len(X_val), 1))
    val_targets = y_val

    for model in models:
        fold_preds = predict_fold(model, val_loader, DEVICE)
        val_preds_accumulator += fold_preds.reshape(-1, 1)

    # Average predictions
    avg_val_preds = val_preds_accumulator / Config.N_FOLDS

    # Calculate Metric
    final_metric = log_loss(val_targets, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # For failure analysis, we need X_val in numpy format which we have
    val_preds = avg_val_preds.flatten()

    # 8. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(val_targets - val_preds)

    # Calculate image statistics for correlation
    # X_val shape: (N, 3, 75, 75) -> (N, C, H, W)
    # Channel 0: Band 1, Channel 1: Band 2
    b1_means = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_val,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 9. Submission
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_loader = get_test_loader(X_test, inc_test, batch_size=BATCH_SIZE)

        test_preds_accumulator = np.zeros((len(X_test), 1))

        for model in models:
            fold_preds = predict_fold(model, test_loader, DEVICE)
            test_preds_accumulator += fold_preds.reshape(-1, 1)

        avg_test_preds = test_preds_accumulator / Config.N_FOLDS
        test_preds = avg_test_preds.flatten()

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
