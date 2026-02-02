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
from library.model import SpatiallyRegularizedSECNN
from library.data_loader import (
    load_and_process_data,
    create_fold_loaders,
    create_test_loader,
)
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
    # We use debug=False to train on full data for best performance within time limit
    # load_cached_data=True to use pre-generated .npy files
    X_train, angles_train, y_train, X_test, angles_test, ids_test = (
        load_and_process_data(debug=Config.DEBUG, load_cached_data=True)
    )

    # 3. Cross-Validation Loop
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    # Array to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(y_train))

    # StratifiedKFold for indices
    # We must use the same seed and parameters as the internal splitter to ensure alignment
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Store fold scores
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nProcessing Fold {fold_idx}...")

        # Create DataLoaders
        train_loader, val_loader = create_fold_loaders(
            X_train, angles_train, y_train, fold_idx
        )

        # Initialize Model
        model = SpatiallyRegularizedSECNN().to(device)

        # Train
        # train_fold returns the best validation loss during training
        train_fold(fold_idx, model, train_loader, val_loader, device)

        # Load Best Model for Inference
        best_checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(best_checkpoint_path, model)

        # Validate on Hold-out set for this fold
        criterion = torch.nn.BCEWithLogitsLoss()
        val_loss, preds, targets = validate(model, val_loader, criterion, device)

        # Store predictions
        # val_loader iterates sequentially over the subset X_train[val_idx]
        oof_preds[val_idx] = preds.flatten()

        fold_score = log_loss(targets, preds)
        fold_scores.append(fold_score)
        print(f"Fold {fold_idx} Log Loss: {fold_score:.6f}")

    # 4. Final Validation Metric
    final_metric = log_loss(y_train, oof_preds)
    # Printing full precision as requested
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_train - oof_preds)

    # Calculate image statistics for correlation
    # X_train is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    # We flatten spatial dims for stats: (N, 3, 5625)
    X_flat = X_train.reshape(X_train.shape[0], 3, -1)

    b1_mean = X_flat[:, 0, :].mean(axis=1)
    b1_std = X_flat[:, 0, :].std(axis=1)
    b2_mean = X_flat[:, 1, :].mean(axis=1)
    b2_std = X_flat[:, 1, :].std(axis=1)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_train,
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

    # 6. Submission Generation
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
            model = SpatiallyRegularizedSECNN().to(device)
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
