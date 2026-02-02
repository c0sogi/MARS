import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders, IcebergDataset
from library.model import MASHCNN
from library.trainer import run_fold


def predict_loader(model, loader, device):
    """
    Runs inference on a DataLoader.
    Returns predictions, targets (if available), and incidence angles.
    """
    model.eval()
    preds = []
    targets = []
    angles = []

    with torch.no_grad():
        for batch in loader:
            # Handle dataset output format: (sample_dict, label) or sample_dict
            if isinstance(batch, (list, tuple)):
                sample = batch[0]
                labels = batch[1]
                targets.extend(labels.numpy())
            else:
                sample = batch

            images = sample["image"].to(device)
            ang = sample["angle"].to(device)

            outputs = model(images, ang)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            angles.extend(ang.cpu().numpy().flatten())

    return np.array(preds), np.array(targets) if targets else None, np.array(angles)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Load all data splits
    train_loader_orig, val_loader_orig, test_loader_orig = get_dataloaders(
        load_cached_data=True
    )

    # Extract underlying data arrays to merge for CV
    X_train = train_loader_orig.dataset.images
    ang_train = train_loader_orig.dataset.angles
    y_train = train_loader_orig.dataset.labels
    ids_train = train_loader_orig.dataset.ids
    trans_train = train_loader_orig.dataset.transform

    X_val = val_loader_orig.dataset.images
    ang_val = val_loader_orig.dataset.angles
    y_val = val_loader_orig.dataset.labels
    ids_val = val_loader_orig.dataset.ids
    trans_val = val_loader_orig.dataset.transform

    # Concatenate Train and Val for Stratified K-Fold
    X_all = np.concatenate([X_train, X_val], axis=0)
    ang_all = np.concatenate([ang_train, ang_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    ids_all = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total labeled samples for CV: {len(y_all)}")

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(y_all))
    oof_targets = np.zeros(len(y_all))
    oof_angles = np.zeros(len(y_all))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n" + "=" * 30)
        print(f"Processing Fold {fold}")
        print("=" * 30)

        # Create Datasets for this fold
        ds_train = IcebergDataset(
            X_all[train_idx],
            ang_all[train_idx],
            y_all[train_idx],
            ids_all[train_idx],
            transform=trans_train,
        )
        ds_val = IcebergDataset(
            X_all[val_idx],
            ang_all[val_idx],
            y_all[val_idx],
            ids_all[val_idx],
            transform=trans_val,
        )

        # Create DataLoaders
        dl_train = DataLoader(
            ds_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Train the model for this fold
        # This function handles the training loop and saves the best checkpoint
        run_fold(fold, dl_train, dl_val)

        # Load the best model for this fold to generate OOF predictions
        print(f"Loading best model for Fold {fold} inference...")
        model = MASHCNN().to(device)
        load_checkpoint(fold, model, device=device, load_best=True)

        # Predict on Validation Fold
        preds, targets, angles = predict_loader(model, dl_val, device)

        # Store OOF predictions
        oof_preds[val_idx] = preds
        oof_targets[val_idx] = targets
        oof_angles[val_idx] = angles

    # 4. Validation & Failure Analysis
    final_metric = log_loss(oof_targets, oof_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Correlation with Incidence Angle
    corr_angle = np.corrcoef(errors, oof_angles)[0, 1]
    print(f"Correlation between Error and Incidence Angle: {corr_angle:.4f}")

    # Correlation with Target Class
    corr_class = np.corrcoef(errors, oof_targets)[0, 1]
    print(f"Correlation between Error and Target Class: {corr_class:.4f}")

    # 5. Submission
    THRESHOLD = 0.18120490171618245

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Initialize array for test predictions
        test_preds_sum = np.zeros(len(test_loader_orig.dataset))

        # Ensemble predictions from all 5 folds
        for fold in range(Config.NUM_FOLDS):
            print(f"Predicting Test Set with Fold {fold} model...")
            model = MASHCNN().to(device)
            load_checkpoint(fold, model, device=device, load_best=True)

            preds, _, _ = predict_loader(model, test_loader_orig, device)
            test_preds_sum += preds

        # Average predictions
        avg_preds = test_preds_sum / Config.NUM_FOLDS

        # Create Submission DataFrame
        test_ids = test_loader_orig.dataset.ids
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
