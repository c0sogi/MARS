import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import load_and_process_data, IcebergDataset
from library.train_eval import run_fold
from library.model import GLPPN


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # X_train corresponds to the data in 'train.csv' (metadata)
    # X_val corresponds to the data in 'val.csv' (metadata)
    # We use X_train for 5-Fold CV training, and X_val for final hold-out validation.
    print("Loading Data...")
    X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, test_ids = (
        load_and_process_data(load_cached_data=True)
    )

    # 3. Stratified 5-Fold Cross-Validation on X_train
    print(
        f"\nStarting Stratified {Config.N_FOLDS}-Fold Cross-Validation on Training Set..."
    )

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    trained_models = []

    # Iterate through folds
    for fold_idx, (train_indices, inner_val_indices) in enumerate(
        skf.split(X_train, y_train)
    ):
        print(f"\n=== Fold {fold_idx} ===")

        # Split data for this fold
        X_fold_train = X_train[train_indices]
        a_fold_train = a_train[train_indices]
        y_fold_train = y_train[train_indices]

        X_fold_val = X_train[inner_val_indices]
        a_fold_val = a_train[inner_val_indices]
        y_fold_val = y_train[inner_val_indices]

        # Create Datasets
        train_ds = IcebergDataset(
            X_fold_train, a_fold_train, y_fold_train, transform=True
        )
        val_ds = IcebergDataset(X_fold_val, a_fold_val, y_fold_val, transform=False)

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

        # Train Model for this fold
        # run_fold returns the best model (loaded with best weights) and the best metric
        model, best_metric = run_fold(fold_idx, train_loader, val_loader)

        # Move model to CPU to save GPU memory and store in list
        model.to("cpu")
        trained_models.append(model)

    # 4. Final Validation on Hold-Out Set (X_val)
    print("\n=== Performing Ensemble Validation on Hold-Out Set ===")

    val_ds_holdout = IcebergDataset(X_val, a_val, y_val, transform=False)
    val_loader_holdout = DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Ensemble Inference
    holdout_preds_accum = np.zeros((len(y_val), 1))

    for i, model in enumerate(trained_models):
        model.to(device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in val_loader_holdout:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        holdout_preds_accum += np.vstack(fold_preds)
        model.to("cpu")  # Move back to CPU

    # Average predictions
    y_pred_val = holdout_preds_accum / Config.N_FOLDS

    # Calculate Metric
    final_metric = calculate_log_loss(y_val, y_pred_val)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude (Log Loss per sample)
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_val, eps, 1 - eps).flatten()
    y_true = y_val.flatten()

    # Per-sample log loss
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Extract features for correlation
    # 1. Incidence Angle (flattened)
    angles_flat = a_val.flatten()

    # 2. Image Intensity Stats (Mean of Band 1 and Band 2)
    # X_val shape: (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    # We compute mean intensity per image.
    band_1_means = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    band_2_means = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    # Create DataFrame
    df_analysis = pd.DataFrame(
        {
            "Error_Magnitude": sample_losses,
            "Incidence_Angle": angles_flat,
            "Band_1_Mean": band_1_means,
            "Band_2_Mean": band_2_means,
        }
    )

    # Compute Correlations
    correlations = df_analysis.corr()["Error_Magnitude"].sort_values(ascending=False)
    print("Correlation of Error Magnitude with Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.17493283735739185

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating Submission..."
        )

        test_ds = IcebergDataset(X_test, a_test, ids=test_ids, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        test_preds_accum = np.zeros((len(test_ids), 1))

        for i, model in enumerate(trained_models):
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs)
                    fold_preds.append(probs.cpu().numpy())

            test_preds_accum += np.vstack(fold_preds)
            model.to("cpu")

        y_pred_test = test_preds_accum / Config.N_FOLDS

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": y_pred_test.flatten()})

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
