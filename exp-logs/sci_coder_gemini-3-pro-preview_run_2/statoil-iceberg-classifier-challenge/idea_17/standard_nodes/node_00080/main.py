import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import WideDualPoolingNet, predict
from library.train_eval import train_fold


def run():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Override Config for Fast Baseline execution
    Config.NUM_EPOCHS = 60
    Config.PATIENCE = 15

    # Setup logger
    logger = setup_logger(os.path.join(Config.WORK_DIR, "run.log"))
    print("Configuration configured for fast baseline.")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    data = process_and_cache_data(load_cached_data=True)

    # Extract Training Data (from train.csv)
    X_train_all = data["X_train"]
    ang_train_all = data["ang_train"]
    y_train_all = data["y_train"]
    ids_train_all = data["ids_train"]

    # Extract Hold-out Validation Data (from val.csv)
    X_holdout = data["X_val"]
    ang_holdout = data["ang_val"]
    y_holdout = data["y_val"]
    ids_holdout = data["ids_val"]

    # Extract Test Data
    X_test = data["X_test"]
    ang_test = data["ang_test"]
    ids_test = data["ids_test"]

    print(f"Training Set Size: {len(X_train_all)}")
    print(f"Hold-out Validation Set Size: {len(X_holdout)}")
    print(f"Test Set Size: {len(X_test)}")

    # 3. Stratified K-Fold Training
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_models = []

    # We train on X_train_all using CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
        print(f"\n--- Starting Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Prepare Fold Data
        train_ds = IcebergDataset(
            X_train_all[train_idx],
            ang_train_all[train_idx],
            y_train_all[train_idx],
            ids_train_all[train_idx],
            transform=True,
        )
        val_ds = IcebergDataset(
            X_train_all[val_idx],
            ang_train_all[val_idx],
            y_train_all[val_idx],
            ids_train_all[val_idx],
            transform=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train
        best_wts, best_loss = train_fold(fold, train_loader, val_loader, Config.DEVICE)

        # Store weights in memory for ensemble
        fold_models.append(best_wts)

        # Clean up
        del train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 4. Hold-out Validation (Ensemble)
    print("\n--- Performing Hold-out Validation ---")
    holdout_ds = IcebergDataset(
        X_holdout, ang_holdout, y_holdout, ids_holdout, transform=False
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    holdout_preds_accum = np.zeros(len(y_holdout))

    for i, wts in enumerate(fold_models):
        model = WideDualPoolingNet().to(Config.DEVICE)
        model.load_state_dict(wts)
        model.eval()

        _, preds = predict(model, holdout_loader, Config.DEVICE)
        holdout_preds_accum += preds

        del model
        torch.cuda.empty_cache()

    avg_holdout_preds = holdout_preds_accum / Config.NUM_FOLDS

    # Calculate Metric
    final_metric = log_loss(y_holdout, avg_holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_holdout - avg_holdout_preds)

    # Compute simple image stats for correlation
    # X_holdout is (N, 3, 75, 75)
    img_means = np.mean(X_holdout, axis=(1, 2, 3))
    img_stds = np.std(X_holdout, axis=(1, 2, 3))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_holdout,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.16676861786296204
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_ds = IcebergDataset(
            X_test, ang_test, labels=None, ids=ids_test, transform=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds_accum = np.zeros(len(ids_test))

        for i, wts in enumerate(fold_models):
            model = WideDualPoolingNet().to(Config.DEVICE)
            model.load_state_dict(wts)
            model.eval()

            _, preds = predict(model, test_loader, Config.DEVICE)
            test_preds_accum += preds

            del model
            torch.cuda.empty_cache()

        avg_test_preds = test_preds_accum / Config.NUM_FOLDS

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
