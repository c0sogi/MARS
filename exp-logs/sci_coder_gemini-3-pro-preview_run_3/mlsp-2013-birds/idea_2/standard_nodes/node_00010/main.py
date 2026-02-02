import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.dataset import BirdDataset, load_dataset_df
from library.train import run_fold, predict_test
from library.utils import seed_everything, calculate_roc_auc, format_and_save_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Adjust epochs for a faster baseline execution while maintaining performance
    Config.EPOCHS = 35

    # 2. Load and Prepare Data
    # Load separate train and val metadata and combine for 5-Fold CV
    df_train_meta = load_dataset_df("train")
    df_val_meta = load_dataset_df("val")
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Load test metadata
    df_test = load_dataset_df("test")

    # Prepare labels for stratification
    num_classes = Config.NUM_CLASSES
    X = df_full["rec_id"].values.reshape(-1, 1)
    y_full = np.zeros((len(df_full), num_classes), dtype=int)

    for idx, row in df_full.iterrows():
        label_str = str(row["labels"])
        if label_str != "?" and label_str.strip() and label_str.lower() != "nan":
            try:
                indices = [int(x) for x in label_str.split()]
                for i in indices:
                    if 0 <= i < num_classes:
                        y_full[idx, i] = 1
            except ValueError:
                pass

    # 3. Cross-Validation Loop
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    oof_preds = np.zeros((len(df_full), num_classes))
    test_preds_sum = np.zeros((len(df_test), num_classes))

    # Prepare Test Loader
    test_dataset = BirdDataset(df_test, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y_full)):
        # Create Fold DataFrames
        train_df = df_full.iloc[train_indices].reset_index(drop=True)
        val_df = df_full.iloc[val_indices].reset_index(drop=True)

        # Train Model for this Fold
        # run_fold returns the best model (loaded from checkpoint) and the best val score
        model, score = run_fold(fold_idx, train_df, val_df, device)

        # Generate OOF Predictions
        # We need to create a loader for the validation set of this fold
        val_dataset = BirdDataset(val_df, phase="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Predict on validation set
        val_preds = predict_test(model, val_loader, device)
        oof_preds[val_indices] = val_preds

        # Predict on Test Set (Accumulate)
        fold_test_preds = predict_test(model, test_loader, device)
        test_preds_sum += fold_test_preds

        # Memory Cleanup
        del model, train_df, val_df, val_dataset, val_loader
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    final_auc = calculate_roc_auc(y_full, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    # Calculate Mean Absolute Error per sample
    errors = np.abs(y_full - oof_preds).mean(axis=1)
    # Calculate Label Count per sample
    num_labels = y_full.sum(axis=1)

    # Calculate Correlation
    if np.std(errors) > 0 and np.std(num_labels) > 0:
        correlation = np.corrcoef(errors, num_labels)[0, 1]
    else:
        correlation = 0.0

    print("Failure Analysis:")
    print(f"Correlation between Error Magnitude and Label Count: {correlation:.10f}")

    # 6. Submission Generation
    threshold = 0.9072993371210134

    if final_auc > threshold:
        avg_test_preds = test_preds_sum / Config.N_FOLDS
        rec_ids = df_test["rec_id"].values
        format_and_save_submission(rec_ids, avg_test_preds, Config.OUTPUT_FILE)
        print(f"Submission saved to {Config.OUTPUT_FILE}")
    else:
        print(
            f"Validation metric {final_auc} did not beat threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
