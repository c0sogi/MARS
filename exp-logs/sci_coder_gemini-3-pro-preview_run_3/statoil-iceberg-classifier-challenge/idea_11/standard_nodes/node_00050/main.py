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
from library.model import SimpleCNN


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

    X_train, angle_train, y_train, ids_train = train_split
    X_val, angle_val, y_val, ids_val = val_split

    # Cite solution_lesson_node_00044: The Pessimism of Out-Of-Fold (OOF) Metrics vs. Hold-Out Ensembles
    # To strictly evaluate against the threshold (which is based on an ensemble hold-out score),
    # we must use the same evaluation strategy: Train on Train Split (using CV for ensemble members)
    # and evaluate the Ensemble on the Validation Split.

    print(f"Training Samples: {len(y_train)}")
    print(f"Hold-Out Validation Samples: {len(y_val)}")

    # --------------------------------------------------------------------------
    # 3. Cross-Validation Loop (on Train Split Only)
    # --------------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for fold_idx, (train_idx, internal_val_idx) in enumerate(
        skf.split(X_train, y_train)
    ):
        print(f"\n--- Running Fold {fold_idx} ---")

        # Split data for this fold (Internal CV split)
        X_tr, X_va = X_train[train_idx], X_train[internal_val_idx]
        angle_tr, angle_va = angle_train[train_idx], angle_train[internal_val_idx]
        y_tr, y_va = y_train[train_idx], y_train[internal_val_idx]
        ids_tr, ids_va = ids_train[train_idx], ids_train[internal_val_idx]

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
        run_fold(fold_idx, train_loader, val_loader)

    # --------------------------------------------------------------------------
    # 4. Validation Metric (Ensemble on Hold-Out Set)
    # --------------------------------------------------------------------------
    print("\n--- Evaluating Ensemble on Hold-Out Validation Set ---")

    # Create Validation Loader (Hold-Out)
    val_ds_holdout = IcebergDataset(
        X_val, angle_val, y_val, ids_val, transform=get_transforms("val")
    )
    val_loader_holdout = torch.utils.data.DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    avg_val_preds = np.zeros((len(y_val),))

    for fold_idx in range(Config.N_FOLDS):
        model = SimpleCNN().to(Config.DEVICE)
        ckpt_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(ckpt_path, model)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angs, _ in val_loader_holdout:
                images = images.to(Config.DEVICE)
                angs = angs.to(Config.DEVICE)

                logits = model(images, angs)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        avg_val_preds += np.concatenate(fold_preds, axis=0).flatten()

    # Average predictions
    avg_val_preds /= Config.N_FOLDS

    final_metric = log_loss(y_val, avg_val_preds)
    print(f"Final Validation Metric (Ensemble Hold-Out): {final_metric}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - avg_val_preds)

    # Correlation with Incidence Angle
    corr_angle, _ = pearsonr(errors, angle_val)
    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")

    # Correlation with Image Statistics
    b1_means = X_val[:, 0, :, :].mean(axis=(1, 2))
    b2_stds = X_val[:, 1, :, :].std(axis=(1, 2))

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

            model = SimpleCNN().to(Config.DEVICE)
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
