import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import _load_and_process_data, IcebergDataset
from library.trainer import train_fold
from library.model import SelectiveSECNN


def run_pipeline():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading data...")
    # Load data using the internal function to get arrays
    # (X_train, ang_train, y_train) corresponds to metadata/train.csv (80%)
    # (X_val, ang_val, y_val) corresponds to metadata/val.csv (20%) - This is our hold-out
    # (X_test, ang_test, ids_test) corresponds to metadata/test.csv
    (train_data, val_data, test_data) = _load_and_process_data(load_cached_data=True)

    X_train_all, ang_train_all, y_train_all = train_data
    X_holdout, ang_holdout, y_holdout = val_data
    X_test, ang_test, ids_test = test_data

    # Debug limit
    if Config.DEBUG:
        limit = Config.DEBUG_SIZE
        print(f"DEBUG MODE: Limiting data to {limit} samples")
        X_train_all = X_train_all[:limit]
        ang_train_all = ang_train_all[:limit]
        y_train_all = y_train_all[:limit]
        X_holdout = X_holdout[:limit]
        ang_holdout = ang_holdout[:limit]
        y_holdout = y_holdout[:limit]
        X_test = X_test[:limit]
        ang_test = ang_test[:limit]
        ids_test = ids_test[:limit]

    # 2. Training (5-Fold CV on the training set)
    # We split X_train_all into 5 folds
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    models = []

    # Define transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation on Training Set...")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_all)):
        print(f"\n--- Fold {fold_idx} ---")

        # Split data
        X_tr, ang_tr, y_tr = (
            X_train_all[train_idx],
            ang_train_all[train_idx],
            y_train_all[train_idx],
        )
        X_va, ang_va, y_va = (
            X_train_all[val_idx],
            ang_train_all[val_idx],
            y_train_all[val_idx],
        )

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Train
        model, best_score = train_fold(fold_idx, train_loader, val_loader)
        models.append(model)

    # 3. Validation on Hold-out Set (Ensemble)
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Create Holdout Loader
    holdout_ds = IcebergDataset(X_holdout, ang_holdout, y_holdout, transform=None)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Inference function
    def predict(loader, model_list):
        # Ensure all models are in eval mode
        for m in model_list:
            m.eval()

        all_preds = []

        with torch.no_grad():
            for batch in loader:
                # Unpack based on length (dataset returns (img, ang, y) or (img, ang))
                if len(batch) == 3:
                    images, angles, _ = batch
                else:
                    images, angles = batch

                images = images.to(device)
                angles = angles.to(device)

                batch_preds = []
                for m in model_list:
                    logits = m(images, angles)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

                # Average predictions across models
                batch_preds = np.array(batch_preds)  # (n_models, batch_size)
                avg_preds = np.mean(batch_preds, axis=0)  # (batch_size,)
                all_preds.append(avg_preds)

        return np.concatenate(all_preds)

    # Predict on holdout
    y_pred_holdout = predict(holdout_loader, models)

    # Calculate Metric
    final_metric = calculate_log_loss(y_holdout, y_pred_holdout)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_holdout - y_pred_holdout)

    # Calculate image stats
    # X_holdout shape: (N, 75, 75, 3)
    # Band 1 is index 0, Band 2 is index 1
    b1_mean = X_holdout[:, :, :, 0].mean(axis=(1, 2))
    b2_mean = X_holdout[:, :, :, 1].mean(axis=(1, 2))
    b1_std = X_holdout[:, :, :, 0].std(axis=(1, 2))
    b2_std = X_holdout[:, :, :, 1].std(axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_holdout,
            "b1_mean": b1_mean,
            "b2_mean": b2_mean,
            "b1_std": b1_std,
            "b2_std": b2_std,
        }
    )

    # Correlation
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation with Error Magnitude:")
    print(correlations.drop("error"))

    # 5. Submission
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Create Test Loader
        test_ds = IcebergDataset(X_test, ang_test, y=None, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Predict
        y_pred_test = predict(test_loader, models)

        # Save
        submission = pd.DataFrame({"id": ids_test, "is_iceberg": y_pred_test})

        submission_path = Config.SUBMISSION_PATH
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
