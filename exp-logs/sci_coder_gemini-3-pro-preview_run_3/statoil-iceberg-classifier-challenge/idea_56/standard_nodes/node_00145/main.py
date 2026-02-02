import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import IcebergDataset, _load_and_process_data
from library.model import DSICNN
from library.train import train_one_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    device = Config.DEVICE

    logger.info("Starting DSI-CNN Pipeline...")

    # 2. Load Data
    # Load cached data (or process if not available)
    data = _load_and_process_data(load_cached_data=True)

    # Merge library's default train/val split to perform 5-Fold CV
    X_total = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_total = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    angles_total = np.concatenate([data["meta_train"], data["meta_val"]], axis=0)

    # Test data
    X_test = data["X_test"]
    angles_test = data["meta_test"]

    # 3. Cross-Validation
    num_folds = Config.NUM_FOLDS
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=Config.SEED)

    # Augmentations for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    oof_preds = np.zeros(len(y_total))
    oof_targets = np.zeros(len(y_total))
    fold_model_paths = []

    # Store indices to map back to original order if needed,
    # but for OOF metric we just need aligned preds/targets
    # We will fill oof_preds using the validation indices

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_total, y_total)):
        logger.info(f"--- Fold {fold_idx} ---")

        # Prepare Fold Data
        X_train_fold, X_val_fold = X_total[train_idx], X_total[val_idx]
        y_train_fold, y_val_fold = y_total[train_idx], y_total[val_idx]
        ang_train_fold, ang_val_fold = angles_total[train_idx], angles_total[val_idx]

        # Datasets
        train_dataset = IcebergDataset(
            X_train_fold, ang_train_fold, y_train_fold, transform=train_transform
        )
        val_dataset = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, transform=None
        )

        # Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        # Model Setup
        model = DSICNN().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold_idx}.pth")

        # We use Config.NUM_EPOCHS. Since dataset is small, 75 epochs is fast.
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    # logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        fold_model_paths.append(best_model_path)

        # Generate OOF predictions for this fold
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # We need raw probabilities for log loss
        # The predict function in library/train.py returns flattened probabilities (sigmoid applied)
        # But it expects a loader without labels usually.
        # We can reuse the predict function but we need a loader that matches its expectation (images, angles)
        # IcebergDataset returns (img, angle, label) if labels are present.
        # The predict function in library/train.py unpacks: `for images, angles in loader:`
        # This will crash if the loader returns 3 items.
        # So we create a temp dataset without labels for prediction.

        val_dataset_no_label = IcebergDataset(
            X_val_fold, ang_val_fold, y=None, transform=None
        )
        val_loader_no_label = DataLoader(
            val_dataset_no_label,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        preds = predict(model, val_loader_no_label, device)

        oof_preds[val_idx] = preds
        oof_targets[val_idx] = y_val_fold

    # 4. Global Evaluation
    # Clip predictions to avoid log(0)
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(oof_targets, oof_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate error magnitude
    errors = np.abs(oof_targets - oof_preds)

    # Calculate features for correlation
    # We use X_total and angles_total.
    # X_total shape: (N, 3, 75, 75). Channels: 0=HH, 1=HV, 2=Avg

    # Mean and Std of Band 1 (HH) and Band 2 (HV)
    b1_mean = np.mean(X_total[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_total[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_total[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_total[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_total,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("\nCorrelation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.17174082291273365
    if final_metric < threshold:
        logger.info(
            f"Metric ({final_metric}) < Threshold ({threshold}). Generating submission..."
        )

        test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        test_preds_accum = np.zeros(len(X_test))

        for model_path in fold_model_paths:
            model = DSICNN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            fold_preds = predict(model, test_loader, device)
            test_preds_accum += fold_preds

        avg_preds = test_preds_accum / num_folds

        # Load test IDs
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)
        submission = pd.DataFrame({"id": df_test_meta["id"], "is_iceberg": avg_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Metric ({final_metric}) >= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
