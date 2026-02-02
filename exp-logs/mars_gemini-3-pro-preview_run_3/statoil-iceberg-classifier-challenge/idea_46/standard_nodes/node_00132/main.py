import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
import torchvision.transforms as T

from library.utils import set_seed, get_device
from library.dataset import load_data, IcebergDataset
from library.trainer import train_fold, predict
from library.model import SPPCNN


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()

    # 2. Load Data
    # load_data handles caching and imputation
    data = load_data(load_cached_data=True)

    # Extract parts
    X_train_part = data["X_train"]
    y_train_part = data["y_train"]
    angle_train_part = data["angle_train"]
    ids_train_part = data["ids_train"]

    X_val_part = data["X_val"]
    y_val_part = data["y_val"]
    angle_val_part = data["angle_val"]
    ids_val_part = data["ids_val"]

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # 3. Combine Train and Val for Stratified K-Fold
    # We want to use all labeled data for training the ensemble
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    angle_full = np.concatenate([angle_train_part, angle_val_part], axis=0)
    ids_full = np.concatenate([ids_train_part, ids_val_part], axis=0)

    # 4. Configure Training
    n_folds = 5
    epochs = 30  # Fast baseline limit
    batch_size = 32
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Arrays to store OOF predictions and Test predictions
    oof_preds_full = np.zeros(len(y_full))
    test_preds_accum = np.zeros(len(ids_test))

    # 5. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        ang_tr, ang_va = angle_full[train_idx], angle_full[val_idx]

        # Define Transforms
        train_transform = T.Compose(
            [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
        )

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Train Fold
        # train_fold saves the best model to checkpoint_dir/model_fold_{fold}.pth
        train_fold(
            fold,
            train_loader,
            val_loader,
            epochs=epochs,
            patience=10,
            checkpoint_dir=checkpoint_dir,
        )

        # Load Best Model for Inference
        model = SPPCNN().to(device)
        model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # OOF Inference (Validation fold)
        # We create a dataset without labels for prediction
        oof_ds = IcebergDataset(X_va, ang_va, y=None, transform=None)
        oof_loader = DataLoader(
            oof_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
        fold_oof_preds = predict(model, oof_loader, device)
        oof_preds_full[val_idx] = fold_oof_preds

        # Test Inference
        test_ds = IcebergDataset(X_test, angle_test, y=None, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

    # 6. Evaluation on Hold-out Validation Set
    # The hold-out set corresponds to the second part of the concatenated arrays (X_val_part)
    # Indices: [len(X_train_part) : ]
    val_start_idx = len(X_train_part)

    # Extract OOF predictions for the specific validation set samples
    val_preds_subset = oof_preds_full[val_start_idx:]
    y_val_subset = y_val_part

    # Compute Metric
    final_metric = log_loss(y_val_subset, val_preds_subset)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Analyze errors on the validation subset
    errors = np.abs(val_preds_subset - y_val_subset)

    # Correlation with Incidence Angle
    # angle_val_part contains imputed values, which is what the model saw
    corr_angle = np.corrcoef(angle_val_part, errors)[0, 1]

    # Correlation with Image Intensity (Band 1 and Band 2 means)
    # X_val_part shape: (N, 3, 75, 75). Channel 0: HH, Channel 1: HV
    mean_b1 = np.mean(X_val_part[:, 0, :, :], axis=(1, 2))
    mean_b2 = np.mean(X_val_part[:, 1, :, :], axis=(1, 2))

    corr_b1 = np.corrcoef(mean_b1, errors)[0, 1]
    corr_b2 = np.corrcoef(mean_b2, errors)[0, 1]

    print("Failure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  Incidence Angle: {corr_angle}")
    print(f"  Mean Band 1 (HH) Intensity: {corr_b1}")
    print(f"  Mean Band 2 (HV) Intensity: {corr_b2}")

    # 8. Submission
    threshold = 0.1806015565870406
    if final_metric < threshold:
        # Average test predictions
        avg_test_preds = test_preds_accum / n_folds

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {final_metric} is not lower than {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
