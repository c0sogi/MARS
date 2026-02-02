import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    NUM_WORKERS,
)
from library.utils import seed_everything, get_device, calculate_score, save_submission
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import WBPA_Net
from library.train import fit_model


def run():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Using device: {device}")

    # 2. Load Data
    # process_and_cache_data returns data split by metadata files (train.csv, val.csv, test.csv)
    # X_train_full corresponds to 'train.csv' -> used for 5-Fold CV
    # X_holdout corresponds to 'val.csv' -> used as strict hold-out for reporting
    (
        X_train_full,
        inc_train_full,
        y_train_full,
        X_holdout,
        inc_holdout,
        y_holdout,
        X_test,
        inc_test,
        ids_test,
    ) = process_and_cache_data(load_cached_data=True)

    print(f"Training Data Shape: {X_train_full.shape}")
    print(f"Hold-out Data Shape: {X_holdout.shape}")

    # 3. Stratified 5-Fold Cross-Validation
    # We train 5 models on the 'train.csv' data
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    models = []

    print(f"\nStarting {n_folds}-Fold Cross-Validation on Training Set...")

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split data for this fold
        X_tr, inc_tr, y_tr = (
            X_train_full[train_idx],
            inc_train_full[train_idx],
            y_train_full[train_idx],
        )
        X_vl, inc_vl, y_vl = (
            X_train_full[val_idx],
            inc_train_full[val_idx],
            y_train_full[val_idx],
        )

        # Create Datasets
        # Apply augmentation only to training set
        train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_vl, inc_vl, y_vl, transform=False)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # Initialize Model
        model = WBPA_Net().to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train
        model = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            scheduler=scheduler,
        )

        models.append(model)

    # 4. Final Validation on Hold-out Set
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Create hold-out loader
    holdout_ds = IcebergDataset(X_holdout, inc_holdout, y_holdout, transform=False)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Ensemble Inference
    ensemble_preds = np.zeros(len(y_holdout))

    with torch.no_grad():
        for i, model in enumerate(models):
            model.eval()
            fold_preds = []
            for inputs, _ in holdout_loader:
                images, angles = inputs
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                fold_preds.extend(probs)

            ensemble_preds += np.array(fold_preds)

    # Average predictions
    ensemble_preds /= n_folds

    # Calculate Metric
    final_metric = calculate_score(y_holdout, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(y_holdout - ensemble_preds)

    # Calculate image statistics for correlation
    # X_holdout shape: (N, 3, 75, 75)
    # Channel 0: HH, Channel 1: HV, Channel 2: Avg
    hh_mean = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    hv_mean = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_holdout,
            "hh_mean": hh_mean,
            "hv_mean": hv_mean,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.17493283735739185
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        test_ds = IcebergDataset(X_test, inc_test, labels=None, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        test_preds = np.zeros(len(ids_test))

        with torch.no_grad():
            for model in models:
                model.eval()
                fold_preds = []
                for inputs in test_loader:
                    # Test loader returns (images, angles) tuple since labels are None
                    images, angles = inputs
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                    fold_preds.extend(probs)

                test_preds += np.array(fold_preds)

        test_preds /= n_folds

        save_submission(ids_test, test_preds, SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
