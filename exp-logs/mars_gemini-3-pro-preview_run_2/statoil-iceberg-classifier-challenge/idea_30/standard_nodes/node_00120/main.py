import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_data, IcebergDataset
from library.model import SC_WBN, predict
from library.train import run_fold


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # load_data returns: X_train, y_train, inc_train, X_val, y_val, inc_val, X_test, inc_test, ids_test
    # Note: X_train/y_train here correspond to the 'train.csv' metadata (for CV)
    #       X_val/y_val correspond to the 'val.csv' metadata (holdout)
    data = load_data(load_cached_data=True)
    X_train_cv, y_train_cv, inc_train_cv = data[0], data[1], data[2]
    X_holdout, y_holdout, inc_holdout = data[3], data[4], data[5]
    X_test, inc_test, ids_test = data[6], data[7], data[8]

    # 3. Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
        # Prepare data for this fold
        X_tr_fold, X_val_fold = X_train_cv[train_idx], X_train_cv[val_idx]
        y_tr_fold, y_val_fold = y_train_cv[train_idx], y_train_cv[val_idx]
        inc_tr_fold, inc_val_fold = inc_train_cv[train_idx], inc_train_cv[val_idx]

        # Execute training for the fold
        # run_fold saves the best model to disk
        run_fold(
            fold,
            X_tr_fold,
            inc_tr_fold,
            y_tr_fold,
            X_val_fold,
            inc_val_fold,
            y_val_fold,
            device,
        )

        # Load the best model for this fold
        model_path = os.path.join(Config.WORKING_DIR, f"sc_wbn_fold_{fold}.pth")
        model = SC_WBN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    # 4. Evaluation on Hold-out Set (Ensemble)
    holdout_dataset = IcebergDataset(X_holdout, inc_holdout, transform=False)
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    holdout_preds_accum = []
    for model in models:
        preds = predict(model, holdout_loader, device)
        holdout_preds_accum.append(preds)

    # Average predictions
    avg_holdout_preds = np.mean(holdout_preds_accum, axis=0)

    # Calculate Metric
    final_metric = log_loss(y_holdout, avg_holdout_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing Failure Analysis on Validation Set...")
    errors = np.abs(y_holdout - avg_holdout_preds)

    # Calculate mean intensity for Band 1 and Band 2 for correlation analysis
    # X_holdout shape: (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_holdout,
            "band_1_mean": b1_means,
            "band_2_mean": b2_means,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = 0.16676861786296204

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds_accum = []
        for model in models:
            preds = predict(model, test_loader, device)
            test_preds_accum.append(preds)

        avg_test_preds = np.mean(test_preds_accum, axis=0)

        submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
