import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data import prepare_data, IcebergDataset, get_transforms
from library.train import run_fold
from library.model import SHMP_CNN


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Limit epochs for a fast baseline execution while ensuring convergence
    Config.NUM_EPOCHS = 25
    set_seed(Config.SEED)

    print(f"Using Device: {Config.DEVICE}")
    print(f"Training for {Config.NUM_EPOCHS} epochs per fold.")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Preparation
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load cached data
    (train_split, val_split, test_data) = prepare_data(load_cached_data=True)

    X_train_split, angle_train_split, y_train_split, ids_train_split = train_split
    X_val_split, angle_val_split, y_val_split, ids_val_split = val_split

    # Merge train and validation splits for 5-Fold Cross-Validation
    # This maximizes the data available for the ensemble
    X = np.concatenate([X_train_split, X_val_split], axis=0)
    angles = np.concatenate([angle_train_split, angle_val_split], axis=0)
    y = np.concatenate([y_train_split, y_val_split], axis=0)
    ids = np.concatenate([ids_train_split, ids_val_split], axis=0)

    print(f"Total training samples for CV: {len(y)}")

    # --------------------------------------------------------------------------
    # 3. Cross-Validation Loop
    # --------------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store Out-Of-Fold (OOF) predictions and targets
    oof_preds = np.zeros(len(y))
    oof_targets = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Running Fold {fold_idx} ---")

        # Split data for this fold
        X_tr, X_va = X[train_idx], X[val_idx]
        angle_tr, angle_va = angles[train_idx], angles[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        ids_tr, ids_va = ids[train_idx], ids[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_tr, angle_tr, y_tr, ids_tr, transform=get_transforms("train")
        )
        val_ds = IcebergDataset(
            X_va, angle_va, y_va, ids_va, transform=get_transforms("val")
        )

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train the model for this fold
        # run_fold handles the training loop and saves the best checkpoint
        run_fold(fold_idx, train_loader, val_loader)

        # ----------------------------------------------------------------------
        # Generate OOF Predictions for this Fold
        # ----------------------------------------------------------------------
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load the best model saved by run_fold
        model = SHMP_CNN().to(Config.DEVICE)
        ckpt_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(ckpt_path, model)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images, angs, _ in val_loader:
                images = images.to(Config.DEVICE)
                angs = angs.to(Config.DEVICE)

                logits = model(images, angs)
                probs = torch.sigmoid(logits)
                fold_probs.append(probs.cpu().numpy())

        # Flatten and store
        fold_probs = np.concatenate(fold_probs, axis=0).flatten()
        oof_preds[val_idx] = fold_probs
        oof_targets[val_idx] = y_va

    # --------------------------------------------------------------------------
    # 4. Validation Metric
    # --------------------------------------------------------------------------
    # Compute Log Loss on the entire OOF set
    final_metric = log_loss(y, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y - oof_preds)

    # Correlation with Incidence Angle
    # Note: angles might have NaNs imputed, but our loaded data has no NaNs
    corr_angle, _ = pearsonr(errors, angles)
    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")

    # Correlation with Image Statistics
    # X shape is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    # We compute mean intensity for HH and Std for HV as representative stats
    b1_means = X[:, 0, :, :].mean(axis=(1, 2))
    b2_stds = X[:, 1, :, :].std(axis=(1, 2))

    corr_b1, _ = pearsonr(errors, b1_means)
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")

    corr_b2, _ = pearsonr(errors, b2_stds)
    print(f"Correlation (Error vs Band 2 Std): {corr_b2:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.18145903282502943

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} passed threshold {threshold}. Generating submission..."
        )

        X_test, angle_test, ids_test = test_data

        # Create Test Dataset and Loader
        test_ds = IcebergDataset(
            X_test, angle_test, y=None, ids=ids_test, transform=get_transforms("test")
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference
        avg_preds = np.zeros((len(ids_test),))

        for fold_idx in range(Config.N_FOLDS):
            print(f"Predicting with model from Fold {fold_idx}...")

            model = SHMP_CNN().to(Config.DEVICE)
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
            )
            load_checkpoint(ckpt_path, model)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angs, _ in test_loader:
                    images = images.to(Config.DEVICE)
                    angs = angs.to(Config.DEVICE)

                    logits = model(images, angs)
                    probs = torch.sigmoid(logits)
                    fold_preds.append(probs.cpu().numpy())

            # Accumulate predictions
            avg_preds += np.concatenate(fold_preds, axis=0).flatten()

        # Average predictions
        avg_preds /= Config.N_FOLDS

        # Save Submission
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
