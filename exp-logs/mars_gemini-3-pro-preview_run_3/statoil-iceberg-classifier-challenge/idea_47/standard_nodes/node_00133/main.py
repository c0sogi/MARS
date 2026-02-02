import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import (
    process_data,
    IcebergDataset,
    BATCH_SIZE,
    METADATA_DIR,
    CHECKPOINT_DIR,
    N_FOLDS,
    SUBMISSION_DIR,
)
from library.model import IDPH_CNN
from library.train import Trainer
from library.utils import seed_everything, get_device


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Data
    # X_full contains both train.csv and val.csv data concatenated
    # We load cached data to save time as per instructions
    X_full, y_full, angle_full, ids_full, X_test, angle_test, ids_test = process_data(
        load_cached_data=True
    )

    # 3. Recover Train/Holdout Split
    # We must separate the holdout set (val.csv) to report the "Final Validation Metric" cleanly.
    # The process_data function concatenated train then val, so we split by length.
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    n_train = len(df_train_meta)

    # CV Data (from train.csv)
    X_cv = X_full[:n_train]
    y_cv = y_full[:n_train]
    angle_cv = angle_full[:n_train]

    # Holdout Data (from val.csv)
    X_holdout = X_full[n_train:]
    y_holdout = y_full[n_train:]
    angle_holdout = angle_full[n_train:]

    print(f"CV Training Data: {X_cv.shape[0]} samples")
    print(f"Holdout Validation Data: {X_holdout.shape[0]} samples")

    # 4. 5-Fold Cross-Validation on CV Data
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Limit epochs for fast baseline execution (override config default of 75)
    FAST_EPOCHS = 35

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\n--- Training Fold {fold_idx + 1}/{N_FOLDS} ---")

        # Fold Data
        X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_val = y_cv[train_idx], y_cv[val_idx]
        a_tr, a_val = angle_cv[train_idx], angle_cv[val_idx]

        # Datasets
        # Apply transforms only to training part of the fold
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

        train_ds = IcebergDataset(X_tr, y_tr, a_tr, transform=train_transform)
        val_ds = IcebergDataset(X_val, y_val, a_val, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Model & Trainer
        model = IDPH_CNN().to(device)
        trainer = Trainer(model, device, fold_idx)

        # Train
        trainer.fit(train_loader, val_loader, epochs=FAST_EPOCHS)

    # 5. Evaluation on Holdout Set (Ensemble)
    print("\n--- Evaluating on Holdout Set ---")
    holdout_ds = IcebergDataset(X_holdout, y_holdout, angle_holdout, transform=None)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    fold_preds = []

    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        model = IDPH_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for imgs, angs, _ in holdout_loader:
                imgs, angs = imgs.to(device), angs.to(device)
                outputs = model(imgs, angs).squeeze(1)
                probs = torch.sigmoid(outputs)
                preds.extend(probs.cpu().numpy())
        fold_preds.append(np.array(preds))

    # Ensemble Average
    avg_preds_holdout = np.mean(fold_preds, axis=0)

    # Metric
    final_metric = log_loss(y_holdout, avg_preds_holdout)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error
    errors = np.abs(y_holdout - avg_preds_holdout)

    # Calculate Features for Correlation
    # We need to compute stats on X_holdout (N, 3, 75, 75)
    # Channel 0: HH, Channel 1: HV
    b1_mean = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_holdout[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angle_holdout,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Correlation
    corr = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation of Error with Features:")
    print(corr)

    # 7. Submission
    TARGET_METRIC = 0.1806015565870406
    if final_metric < TARGET_METRIC:
        print("\nMetric passed threshold. Generating submission...")

        test_ds = IcebergDataset(X_test, None, angle_test, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_fold_preds = []
        for fold_idx in range(N_FOLDS):
            model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
            model = IDPH_CNN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            preds = []
            with torch.no_grad():
                for imgs, angs in test_loader:
                    imgs, angs = imgs.to(device), angs.to(device)
                    outputs = model(imgs, angs).squeeze(1)
                    probs = torch.sigmoid(outputs)
                    preds.extend(probs.cpu().numpy())
            test_fold_preds.append(np.array(preds))

        avg_preds_test = np.mean(test_fold_preds, axis=0)

        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds_test})
        out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
