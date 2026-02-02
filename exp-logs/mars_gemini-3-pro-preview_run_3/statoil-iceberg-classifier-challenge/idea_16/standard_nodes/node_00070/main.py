import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from sklearn.model_selection import train_test_split
from library.model import SimpleCNN
from library.data_loader import (
    load_and_process_data,
    create_fold_loaders,
    create_test_loader,
    IcebergDataset,
)
from torch.utils.data import DataLoader
from library.trainer import train_fold, validate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    print("Loading and processing data...")
    X_train, angles_train, y_train, X_test, angles_test, ids_test = (
        load_and_process_data(debug=Config.DEBUG, load_cached_data=True)
    )

    # 3. Hold-Out Split Strategy (Cite Lesson 44)
    # We reserve 20% of the data for the final ensemble evaluation to match the "Current Best" methodology.
    print("Splitting data into Development (80%) and Hold-Out (20%)...")
    X_dev, X_holdout, angles_dev, angles_holdout, y_dev, y_holdout = train_test_split(
        X_train,
        angles_train,
        y_train,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=y_train,
    )

    # 4. Cross-Validation Loop on Development Set
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation on Development Set...")

    # StratifiedKFold for indices on X_dev
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        print(f"\nProcessing Fold {fold_idx}...")

        # Create DataLoaders using the subset X_dev
        train_loader, val_loader = create_fold_loaders(
            X_dev, angles_dev, y_dev, fold_idx
        )

        # Initialize Model (SimpleCNN)
        model = SimpleCNN().to(device)

        # Train
        train_fold(fold_idx, model, train_loader, val_loader, device)

    # 5. Hold-Out Ensemble Evaluation
    print(f"\nEvaluating Ensemble on Hold-Out Set ({len(X_holdout)} samples)...")

    # Create Hold-Out Loader
    holdout_dataset = IcebergDataset(
        X_holdout, angles_holdout, y=y_holdout, transform=None
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Collect predictions from all 5 models
    holdout_preds_matrix = np.zeros((len(X_holdout), Config.NUM_FOLDS))

    for fold_idx in range(Config.NUM_FOLDS):
        model = SimpleCNN().to(device)
        best_checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(best_checkpoint_path, model)

        criterion = torch.nn.BCEWithLogitsLoss()
        _, preds, _ = validate(model, holdout_loader, criterion, device)
        holdout_preds_matrix[:, fold_idx] = preds.flatten()

    # Average predictions (Ensemble)
    ensemble_preds = holdout_preds_matrix.mean(axis=1)

    # 6. Final Validation Metric
    final_metric = log_loss(y_holdout, ensemble_preds)
    # Printing full precision as requested
    print(f"\nFinal Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error on Hold-Out set
    errors = np.abs(y_holdout - ensemble_preds)

    # Calculate image statistics for correlation on Hold-Out set
    X_flat = X_holdout.reshape(X_holdout.shape[0], 3, -1)

    b1_mean = X_flat[:, 0, :].mean(axis=1)
    b1_std = X_flat[:, 0, :].std(axis=1)
    b2_mean = X_flat[:, 1, :].mean(axis=1)
    b2_std = X_flat[:, 1, :].std(axis=1)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_holdout,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 8. Submission Generation
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({threshold}). Generating submission..."
        )

        test_loader = create_test_loader(X_test, angles_test, ids_test)

        # Array to store predictions from each fold: (N_test, N_folds)
        fold_test_preds = np.zeros((len(X_test), Config.NUM_FOLDS))

        for fold_idx in range(Config.NUM_FOLDS):
            print(f"Predicting with Fold {fold_idx} model...")
            model = SimpleCNN().to(device)
            best_checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
            )
            load_checkpoint(best_checkpoint_path, model)

            # Inference
            model.eval()
            fold_preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)
                    logits = model(images, angles)
                    probs = torch.sigmoid(logits)
                    fold_preds.append(probs.cpu().numpy())

            fold_test_preds[:, fold_idx] = np.concatenate(fold_preds).flatten()

        # Average predictions (Ensemble)
        avg_preds = fold_test_preds.mean(axis=1)

        # Create submission DataFrame
        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

        # Save
        save_path = "./submission/submission.csv"
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
