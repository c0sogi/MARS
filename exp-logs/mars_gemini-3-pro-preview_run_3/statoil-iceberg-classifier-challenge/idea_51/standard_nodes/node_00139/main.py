import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, log_loss_score
from library.dataset import process_data, IcebergDataset, get_median_angle
from library.model import DPDCNN
from library.train import train_one_fold


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use 'train.csv' data for the Cross-Validation loop.
    # We use 'val.csv' data as the Hold-out set for final evaluation.
    # We use 'test.csv' data for submission.

    # Get median angle for imputation (calculated from train metadata)
    angle_median = get_median_angle()

    print("Loading datasets...")
    X_train_cv, ang_train_cv, y_train_cv, ids_train_cv = process_data(
        Config.TRAIN_META_PATH,
        Config.TRAIN_JSON,
        "train",
        angle_median,
        load_cached_data=True,
    )

    X_holdout, ang_holdout, y_holdout, ids_holdout = process_data(
        Config.VAL_META_PATH,
        Config.TRAIN_JSON,
        "val",
        angle_median,
        load_cached_data=True,
    )

    X_test, ang_test, _, ids_test = process_data(
        Config.TEST_META_PATH,
        Config.TEST_JSON,
        "test",
        angle_median,
        load_cached_data=True,
    )

    # 3. 5-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    fold_models = []

    # Define augmentations for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    print(
        f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation on {len(X_train_cv)} samples..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
        print(f"\n--- Fold {fold} ---")

        # Split data for this fold
        X_tr, ang_tr, y_tr = (
            X_train_cv[train_idx],
            ang_train_cv[train_idx],
            y_train_cv[train_idx],
        )
        X_val, ang_val, y_val = (
            X_train_cv[val_idx],
            ang_train_cv[val_idx],
            y_train_cv[val_idx],
        )

        # Create Datasets
        ds_tr = IcebergDataset(
            X_tr, ang_tr, y_tr, transform=train_transform, mode="train"
        )
        ds_val = IcebergDataset(X_val, ang_val, y_val, transform=None, mode="val")

        # Create DataLoaders
        # Pin memory speeds up host-to-device transfer
        pin = Config.DEVICE == "cuda"
        dl_tr = DataLoader(
            ds_tr,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=pin,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=pin,
        )

        # Train the fold
        best_state, _, _ = train_one_fold(fold, dl_tr, dl_val)
        fold_models.append(best_state)

    # 4. Hold-out Validation (Ensemble Inference)
    print("\nRunning inference on Hold-out Validation Set...")

    ds_holdout = IcebergDataset(
        X_holdout, ang_holdout, y_holdout, transform=None, mode="val"
    )
    dl_holdout = DataLoader(
        ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Accumulate predictions from all folds
    avg_preds_holdout = np.zeros(len(y_holdout))

    for i, state in enumerate(fold_models):
        model = DPDCNN()
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for images, angles, _ in dl_holdout:
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                preds.append(probs.cpu().numpy())

        # Add to ensemble average
        avg_preds_holdout += np.concatenate(preds).ravel() / Config.N_FOLDS

    final_metric = log_loss_score(y_holdout, avg_preds_holdout)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(y_holdout - avg_preds_holdout)

    # Extract simple features from the holdout set for correlation analysis
    # X_holdout shape: (N, 3, 75, 75). Channel 0: HH, Channel 1: HV
    b1_mean = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_holdout[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))
    b2_max = np.max(X_holdout[:, 1, :, :], axis=(1, 2))

    features = {
        "inc_angle": ang_holdout,
        "b1_mean": b1_mean,
        "b1_std": b1_std,
        "b2_mean": b2_mean,
        "b2_max": b2_max,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, feat in features.items():
        # Calculate correlation (index [0, 1] of the correlation matrix)
        corr = np.corrcoef(errors, feat)[0, 1]
        print(f"  {name}: {corr:.4f}")

    # 6. Submission
    threshold = 0.17174082291273365
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) < Threshold ({threshold}). Generating submission..."
        )

        ds_test = IcebergDataset(
            X_test, ang_test, ids=ids_test, transform=None, mode="test"
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        avg_preds_test = np.zeros(len(ids_test))

        for i, state in enumerate(fold_models):
            model = DPDCNN()
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            preds = []
            with torch.no_grad():
                for images, angles, _ in dl_test:
                    images = images.to(device)
                    angles = angles.to(device)
                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs)
                    preds.append(probs.cpu().numpy())

            avg_preds_test += np.concatenate(preds).ravel() / Config.N_FOLDS

        submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds_test})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
