import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torchvision import transforms

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import _get_data_splits, IcebergDataset
from library.model import PDPH_SE_CNN
from library.train import train_one_epoch, validate, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline execution while maintaining performance
    Config.DEBUG = False
    Config.NUM_EPOCHS = 50  # Sufficient for convergence on small dataset
    Config.BATCH_SIZE = 32

    # Setup
    logger = setup_logger()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Starting Runfile Execution")
    logger.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load data from cache (or process if needed)
    data_map = _get_data_splits(load_cached_data=True)

    # Training Data (from train.csv)
    X_train = data_map["X_train"]
    y_train = data_map["y_train"]
    ang_train = data_map["angles_train"]
    ids_train = data_map["ids_train"]

    # Hold-out Validation Data (from val.csv)
    X_val = data_map["X_val"]
    y_val = data_map["y_val"]
    ang_val = data_map["angles_val"]
    ids_val = data_map["ids_val"]

    # Test Data (from test.csv)
    X_test = data_map["X_test"]
    ang_test = data_map["angles_test"]
    ids_test = data_map["ids_test"]

    logger.info(f"Train Set: {len(X_train)}")
    logger.info(f"Hold-out Val Set: {len(X_val)}")
    logger.info(f"Test Set: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 3. Training Loop (5-Fold CV on Train Set)
    # -------------------------------------------------------------------------
    # We train an ensemble of 5 models on the Train Set.
    # We use K-Fold internally on the Train Set to monitor convergence and select models.

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Directory to store models
    checkpoints_dir = os.path.join(Config.WORKING_DIR, "runfile_checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Augmentation
    train_transform = None
    if Config.USE_AUGMENTATION:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.VERTICAL_FLIP_PROB),
            ]
        )

    trained_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"\n--- Training Fold {fold + 1}/{n_folds} ---")

        # Split internal train/val
        X_tr_fold, X_va_fold = X_train[tr_idx], X_train[va_idx]
        y_tr_fold, y_va_fold = y_train[tr_idx], y_train[va_idx]
        ang_tr_fold, ang_va_fold = ang_train[tr_idx], ang_train[va_idx]
        ids_tr_fold, ids_va_fold = ids_train[tr_idx], ids_train[va_idx]

        # Datasets & Loaders
        train_ds = IcebergDataset(
            X_tr_fold,
            ang_tr_fold,
            y_tr_fold,
            transform=train_transform,
            ids=ids_tr_fold,
        )
        val_ds = IcebergDataset(
            X_va_fold, ang_va_fold, y_va_fold, transform=None, ids=ids_va_fold
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Model, Criterion, Optimizer
        model = PDPH_SE_CNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        best_loss = float("inf")
        patience = 0
        model_path = os.path.join(checkpoints_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, _, _ = validate(model, val_loader, criterion, device)

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience += 1
                if patience >= Config.PATIENCE:
                    break

        logger.info(f"Fold {fold+1} Best Internal Val Loss: {best_loss:.6f}")
        trained_models.append(model_path)

    # -------------------------------------------------------------------------
    # 4. Validation on Hold-out Set
    # -------------------------------------------------------------------------
    logger.info("\n--- Evaluating on Hold-out Validation Set ---")

    val_dataset = IcebergDataset(X_val, ang_val, y=None, transform=None, ids=ids_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    fold_preds = []

    for model_path in trained_models:
        model = PDPH_SE_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        preds = predict(model, val_loader, device)
        fold_preds.append(preds)

    # Ensemble Average
    avg_val_preds = np.mean(fold_preds, axis=0)

    # Clip for stability
    avg_val_preds_clipped = np.clip(avg_val_preds, 1e-15, 1 - 1e-15)

    # Metric Calculation
    final_metric = log_loss(y_val, avg_val_preds_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(y_val - avg_val_preds)

    # Extract Features for correlation
    # Image stats (Band 1 and Band 2)
    # X_val shape: (N, 75, 75, 3) -> 0:HH, 1:HV
    b1_mean = np.mean(X_val[..., 0], axis=(1, 2))
    b1_std = np.std(X_val[..., 0], axis=(1, 2))
    b1_max = np.max(X_val[..., 0], axis=(1, 2))
    b1_min = np.min(X_val[..., 0], axis=(1, 2))

    b2_mean = np.mean(X_val[..., 1], axis=(1, 2))
    b2_std = np.std(X_val[..., 1], axis=(1, 2))
    b2_max = np.max(X_val[..., 1], axis=(1, 2))
    b2_min = np.min(X_val[..., 1], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b1_max": b1_max,
            "b1_min": b1_min,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
            "b2_max": b2_max,
            "b2_min": b2_min,
        }
    )

    # Correlation
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("error"))

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        logger.info(
            f"\nMetric ({final_metric:.6f}) < Threshold ({THRESHOLD:.6f}). Generating Submission..."
        )

        test_dataset = IcebergDataset(
            X_test, ang_test, y=None, transform=None, ids=ids_test
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        test_fold_preds = []
        for model_path in trained_models:
            model = PDPH_SE_CNN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            preds = predict(model, test_loader, device)
            test_fold_preds.append(preds)

        avg_test_preds = np.mean(test_fold_preds, axis=0)

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"\nMetric ({final_metric:.6f}) >= Threshold ({THRESHOLD:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
