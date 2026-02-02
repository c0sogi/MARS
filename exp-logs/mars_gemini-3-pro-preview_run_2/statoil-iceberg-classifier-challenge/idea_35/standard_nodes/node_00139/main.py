import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_global_stats
from library.data_loader import process_data, IcebergDataset, get_transforms
from library.model import RIWBN
from library.trainer import train_one_epoch, validate, predict, EarlyStopping


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # "Fast Baseline" configuration overrides
    # We use 50 epochs which is sufficient for this small dataset (~1.6k images)
    # to converge while keeping runtime short.
    EPOCHS = 50
    BATCH_SIZE = Config.BATCH_SIZE
    DEVICE = Config.DEVICE

    print(f"Running on device: {DEVICE}")

    # 2. Data Preparation
    # Load cached processed data
    # This returns a dictionary with X_train, X_val, X_test, etc.
    data = process_data(load_cached_data=True)

    # Load global stats for Band 1 and Band 2
    stats = calculate_global_stats(load_cached_data=True, debug=Config.DEBUG)

    # Combine Train and Validation sets for Stratified K-Fold
    # We merge them to perform our own stratified split
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    inc_full = np.concatenate([data["inc_train"], data["inc_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    # Compute statistics for the 3rd channel (Mean of B1 and B2)
    # This is required for normalization in the Dataset class
    b3_data = X_full[:, 2, :, :]
    stats["b3_min"] = float(b3_data.min())
    stats["b3_max"] = float(b3_data.max())

    # Prepare Test Dataset and Loader
    test_dataset = IcebergDataset(
        X=data["X_test"],
        inc_angles=data["inc_test"],
        labels=None,
        transform=get_transforms("test"),
        global_stats=stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    # 3. Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store results
    oof_preds = np.zeros(len(y_full))
    test_preds_accum = np.zeros(len(test_dataset))

    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Split Data
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        inc_tr, inc_val = inc_full[train_idx], inc_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_tr, inc_tr, y_tr, transform=get_transforms("train"), global_stats=stats
        )
        val_ds = IcebergDataset(
            X_val, inc_val, y_val, transform=get_transforms("val"), global_stats=stats
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
        )

        # Initialize Model
        model = RIWBN().to(DEVICE)

        # Optimizer & Scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        es = EarlyStopping(patience=Config.PATIENCE, verbose=False)

        # Training Loop
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)
            es(val_loss, model)

            if es.early_stop:
                break

        # Load Best Weights
        if es.best_state:
            model.load_state_dict(es.best_state)

        # Save Model Checkpoint
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth"),
        )

        # Generate OOF Predictions (Probabilities)
        val_probs = predict(model, val_loader, DEVICE)
        oof_preds[val_idx] = val_probs

        # Generate Test Predictions (Probabilities)
        fold_test_probs = predict(model, test_loader, DEVICE)
        test_preds_accum += fold_test_probs

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 4. Validation Metric Calculation
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error magnitude
    errors = np.abs(y_full - oof_preds)

    # Calculate feature statistics for correlation analysis
    # Flatten spatial dims: (N, 3, 75, 75) -> (N, 3, 5625)
    X_flat = X_full.reshape(X_full.shape[0], 3, -1)

    # Extract basic stats from raw data (Band 1 and Band 2)
    b1_mean = np.mean(X_flat[:, 0, :], axis=1)
    b1_std = np.std(X_flat[:, 0, :], axis=1)
    b2_mean = np.mean(X_flat[:, 1, :], axis=1)
    b2_std = np.std(X_flat[:, 1, :], axis=1)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_full,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute and print correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Average predictions across folds
        avg_test_preds = test_preds_accum / Config.NUM_FOLDS

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"id": data["ids_test"], "is_iceberg": avg_test_preds}
        )

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
