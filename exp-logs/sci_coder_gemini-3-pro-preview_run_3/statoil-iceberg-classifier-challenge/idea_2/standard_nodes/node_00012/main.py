import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import (
    TRAIN_JSON,
    TRAIN_META,
    VAL_META,
    SUBMISSION_FILE,
    WORKING_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
)
from library.utils import seed_everything
from library.data_loader import load_and_process_data, get_test_loader
from library.model import IcebergCNN, predict
from library.train import run_fold


def main():
    # Set seed for reproducibility
    seed_everything(SEED)

    # ---------------------------------------------------------
    # 1. Data Loading and Splitting
    # ---------------------------------------------------------
    print("Loading and processing data...")
    # Load full training data and test data
    # X_full corresponds to the entire train.json
    X_full, y_full, angles_full, X_test, ids_test, angles_test = load_and_process_data(
        load_cached_data=True
    )

    # Load raw train.json to map IDs to indices (since load_and_process_data doesn't return train IDs)
    with open(TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    # Create a mapping from ID to index
    id_to_idx = {item["id"]: i for i, item in enumerate(raw_train_data)}

    # Load Metadata for splitting
    df_train_meta = pd.read_csv(TRAIN_META)
    df_val_meta = pd.read_csv(VAL_META)

    # Get indices for Train and Hold-out Validation sets
    train_indices = [id_to_idx[uid] for uid in df_train_meta["id"] if uid in id_to_idx]
    val_indices = [id_to_idx[uid] for uid in df_val_meta["id"] if uid in id_to_idx]

    # Create the subsets
    X_train_subset = X_full[train_indices]
    y_train_subset = y_full[train_indices]
    angles_train_subset = angles_full[train_indices]

    X_val_holdout = X_full[val_indices]
    y_val_holdout = y_full[val_indices]
    angles_val_holdout = angles_full[val_indices]

    print(f"Training Subset Size: {len(X_train_subset)}")
    print(f"Hold-out Validation Size: {len(X_val_holdout)}")

    # ---------------------------------------------------------
    # 2. Training (5-Fold CV on Training Subset)
    # ---------------------------------------------------------
    n_folds = 5
    # Use fewer epochs for the fast baseline requirement, but enough to converge
    epochs_per_fold = 20

    trained_models = []

    print(f"Starting {n_folds}-Fold Cross-Validation on Training Subset...")

    for fold in range(n_folds):
        # run_fold saves checkpoints to WORKING_DIR/fold_{fold}
        # It uses StratifiedKFold internally on the provided data to create a train/dev split for early stopping
        run_fold(
            fold_idx=fold,
            X_train=X_train_subset,
            y_train=y_train_subset,
            angles_train=angles_train_subset,
            num_epochs=epochs_per_fold,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )

        # Load the best model from this fold
        fold_checkpoint_dir = os.path.join(WORKING_DIR, f"fold_{fold}")
        best_model_path = os.path.join(fold_checkpoint_dir, "model_best.pth")

        model = IcebergCNN(dropout_rate=0.5).to(DEVICE)
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        trained_models.append(model)

    # ---------------------------------------------------------
    # 3. Validation Assessment
    # ---------------------------------------------------------
    print("Evaluating on Hold-out Validation Set...")

    # Create DataLoader for hold-out set
    val_loader = get_test_loader(
        X_val_holdout, angles_val_holdout, batch_size=BATCH_SIZE
    )

    # Ensemble predictions
    val_preds_accum = np.zeros(len(X_val_holdout))

    for model in trained_models:
        preds = predict(val_loader, model, DEVICE)
        val_preds_accum += preds

    avg_val_preds = val_preds_accum / n_folds

    # Calculate Metric
    # Clip to avoid log(0) errors, though model outputs sigmoid so it's (0,1)
    avg_val_preds_clipped = np.clip(avg_val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_val_holdout, avg_val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_val_holdout - avg_val_preds)

    # Correlation with Incidence Angle
    corr_angle, _ = pearsonr(errors, angles_val_holdout)
    print(f"Correlation between Error and Incidence Angle: {corr_angle}")

    # Correlation with Image Signal Strength (Mean of Band 1)
    # X_val_holdout is (N, 224, 224, 3). Band 1 is index 0.
    # Note: Data is scaled to [0, 1] roughly. We use this as proxy for signal strength.
    band1_means = np.mean(X_val_holdout[:, :, :, 0], axis=(1, 2))
    corr_band1, _ = pearsonr(errors, band1_means)
    print(f"Correlation between Error and Band 1 Mean: {corr_band1}")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    threshold = 0.2089132981339209

    if final_metric < threshold:
        print("Metric passed threshold. Generating submission...")

        test_loader = get_test_loader(X_test, angles_test, batch_size=BATCH_SIZE)
        test_preds_accum = np.zeros(len(X_test))

        for model in trained_models:
            preds = predict(test_loader, model, DEVICE)
            test_preds_accum += preds

        avg_test_preds = test_preds_accum / n_folds

        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})
        df_sub.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved successfully to: {SUBMISSION_FILE}")
    else:
        print(
            f"Metric {final_metric} did not pass threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
