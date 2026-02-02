import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_global_stats
from library.model import IcebergDataset, NFWBN, get_inc_angle_stats
from library.train_eval import run_fold


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    Config.NUM_EPOCHS = 20  # Reduced from 100 for speed
    Config.NUM_FOLDS = 5  # Standard 5-fold
    Config.DEBUG = False

    print(f"Starting execution on device: {Config.DEVICE}")
    print(f"Work Directory: {Config.WORK_DIR}")

    # Ensure work directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    fold_dir = os.path.join(Config.WORK_DIR, "folds")
    os.makedirs(fold_dir, exist_ok=True)

    # 2. Prepare Statistics
    # Calculate global image stats (min/max) for scaling
    global_stats = calculate_global_stats(load_cached_data=True)

    # Calculate incidence angle stats from the training metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    inc_angle_stats = get_inc_angle_stats(train_meta_path)

    # 3. Prepare Training Data (5-Fold CV on metadata/train.csv)
    # We use metadata/train.csv for CV, keeping metadata/val.csv as pure hold-out
    df_train_full = pd.read_csv(train_meta_path)

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 4. Training Loop
    print("\n=== Starting 5-Fold Cross-Validation Training ===")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["is_iceberg"])
    ):
        print(f"\n--- Fold {fold} ---")

        # Create Fold CSVs
        train_fold_df = df_train_full.iloc[train_idx]
        val_fold_df = df_train_full.iloc[val_idx]

        fold_train_path = os.path.join(fold_dir, f"train_fold_{fold}.csv")
        fold_val_path = os.path.join(fold_dir, f"val_fold_{fold}.csv")

        train_fold_df.to_csv(fold_train_path, index=False)
        val_fold_df.to_csv(fold_val_path, index=False)

        # Create Datasets
        train_ds = IcebergDataset(
            fold_train_path,
            os.path.join(Config.INPUT_DIR, "train.json"),
            transform=True,
            global_stats=global_stats,
            inc_angle_stats=inc_angle_stats,
        )

        val_ds = IcebergDataset(
            fold_val_path,
            os.path.join(Config.INPUT_DIR, "train.json"),
            transform=False,
            global_stats=global_stats,
            inc_angle_stats=inc_angle_stats,
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Run Training for this fold
        run_fold(fold, train_loader, val_loader)

    # 5. Hold-out Validation
    print("\n=== Hold-out Validation Evaluation ===")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")

    val_holdout_ds = IcebergDataset(
        val_meta_path,
        os.path.join(Config.INPUT_DIR, "train.json"),
        transform=False,
        global_stats=global_stats,
        inc_angle_stats=inc_angle_stats,
    )

    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Ensemble Prediction on Hold-out
    all_preds = np.zeros((len(val_holdout_ds), Config.NUM_FOLDS))
    targets_list = []

    # Collect targets once
    for _, _, targets in val_holdout_loader:
        targets_list.extend(targets.numpy().flatten())
    y_true = np.array(targets_list)

    # Predict with each fold model
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        model = NFWBN().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for imgs, angles, _ in val_holdout_loader:
                imgs = imgs.to(Config.DEVICE)
                angles = angles.to(Config.DEVICE)
                logits = model(imgs, angles)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_preds.extend(probs.flatten())

        all_preds[:, fold] = fold_preds

    # Average predictions
    y_pred = np.mean(all_preds, axis=1)

    # Compute Metric
    final_metric = log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred)

    # Collect features for correlation analysis
    inc_angles = []
    img_means = []
    img_stds = []

    # We need to iterate dataset to get raw values (or processed ones)
    # Using the loader is efficient
    for imgs, angles, _ in val_holdout_loader:
        # angles are normalized, but correlation is scale invariant so it's fine
        inc_angles.extend(angles.numpy().flatten())

        # Calculate image stats from the tensor (3 channels: B1, B2, Avg)
        # We'll use the mean of the first channel (Band 1) as a proxy for intensity
        b1 = imgs[:, 0, :, :]
        img_means.extend(torch.mean(b1, dim=(1, 2)).numpy().flatten())
        img_stds.extend(torch.std(b1, dim=(1, 2)).numpy().flatten())

    inc_angles = np.array(inc_angles)
    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Compute correlations
    # Handle NaN in inc_angle if any (though dataset imputes them, let's be safe)
    valid_idx = ~np.isnan(inc_angles)

    if np.sum(valid_idx) > 1:
        corr_angle, _ = pearsonr(errors[valid_idx], inc_angles[valid_idx])
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (Insufficient data)")

    corr_mean, _ = pearsonr(errors, img_means)
    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean:.4f}")

    corr_std, _ = pearsonr(errors, img_stds)
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std:.4f}")

    # 7. Submission
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_ds = IcebergDataset(
            test_meta_path,
            os.path.join(Config.INPUT_DIR, "test.json"),
            transform=False,
            global_stats=global_stats,
            inc_angle_stats=inc_angle_stats,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        test_preds = np.zeros((len(test_ds), Config.NUM_FOLDS))
        ids = []

        # Collect IDs
        for _, _, img_ids in test_loader:
            ids.extend(img_ids)

        # Predict
        for fold in range(Config.NUM_FOLDS):
            print(f"Predicting Test Set with Fold {fold}...")
            model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
            model = NFWBN().to(Config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for imgs, angles, _ in test_loader:
                    imgs = imgs.to(Config.DEVICE)
                    angles = angles.to(Config.DEVICE)
                    logits = model(imgs, angles)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_preds.extend(probs.flatten())

            test_preds[:, fold] = fold_preds

        avg_test_preds = np.mean(test_preds, axis=1)

        # Save
        df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_test_preds})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_metric} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
