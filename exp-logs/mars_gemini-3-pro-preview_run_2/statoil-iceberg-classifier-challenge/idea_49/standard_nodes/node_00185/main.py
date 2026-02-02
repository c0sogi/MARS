import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.data_loader import process_data, IcebergDataset
from library.model import RDPWBN
from library.train_eval import train_fold
from library.utils import setup_logger


def run():
    # 1. Setup
    Config.setup()
    logger = setup_logger(os.path.join(Config.WORK_DIR, "run.log"))
    logger.info("Starting runfile.py execution...")

    # 2. Data Loading
    # Load metadata to define the Train/Validation split
    logger.info("Loading metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Load processed data (images, angles, labels)
    logger.info("Loading and processing image data...")
    train_data_full, test_data_full, global_stats = process_data(load_cached_data=True)

    # Map IDs to indices in the loaded numpy arrays
    id_to_idx = {uid: i for i, uid in enumerate(train_data_full["ids"])}

    # Identify indices for Meta-Train (for CV training) and Meta-Val (for hold-out evaluation)
    train_indices = [id_to_idx[uid] for uid in df_train_meta["id"] if uid in id_to_idx]
    val_indices = [id_to_idx[uid] for uid in df_val_meta["id"] if uid in id_to_idx]

    # Create data arrays for Meta-Train
    X_train = train_data_full["images"][train_indices]
    y_train = train_data_full["labels"][train_indices]
    ang_train = train_data_full["angles"][train_indices]
    ids_train = train_data_full["ids"][train_indices]

    # Create data arrays for Meta-Val (Hold-out)
    X_holdout = train_data_full["images"][val_indices]
    y_holdout = train_data_full["labels"][val_indices]
    ang_holdout = train_data_full["angles"][val_indices]
    ids_holdout = train_data_full["ids"][val_indices]

    logger.info(f"Meta-Train size: {len(X_train)}")
    logger.info(f"Meta-Val (Hold-out) size: {len(X_holdout)}")

    # 3. Training: Stratified 5-Fold Cross-Validation on Meta-Train
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    trained_models = []

    # Loop through folds
    for fold, (inner_train_idx, inner_val_idx) in enumerate(
        skf.split(X_train, y_train)
    ):
        logger.info(f"--- Starting Fold {fold} ---")

        # Create Datasets for this fold
        ds_fold_train = IcebergDataset(
            X_train[inner_train_idx],
            ang_train[inner_train_idx],
            y_train[inner_train_idx],
            ids_train[inner_train_idx],
            transform=True,
            global_stats=global_stats,
        )
        ds_fold_val = IcebergDataset(
            X_train[inner_val_idx],
            ang_train[inner_val_idx],
            y_train[inner_val_idx],
            ids_train[inner_val_idx],
            transform=False,
            global_stats=global_stats,
        )

        # Create DataLoaders
        dl_fold_train = DataLoader(
            ds_fold_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )
        dl_fold_val = DataLoader(
            ds_fold_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Train the model for this fold
        best_state = train_fold(fold, dl_fold_train, dl_fold_val, logger)

        # Load the best state into a new model instance for inference later
        model = RDPWBN().to(Config.DEVICE)
        model.load_state_dict(best_state)
        model.eval()
        trained_models.append(model)

        # Clean up to save memory
        del ds_fold_train, ds_fold_val, dl_fold_train, dl_fold_val
        torch.cuda.empty_cache()

    # 4. Validation: Ensemble Inference on Hold-out Set
    logger.info("Performing ensemble inference on hold-out validation set...")

    ds_holdout = IcebergDataset(
        X_holdout,
        ang_holdout,
        y_holdout,
        ids_holdout,
        transform=False,
        global_stats=global_stats,
    )
    dl_holdout = DataLoader(
        ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Collect predictions from all models
    ensemble_preds = np.zeros((len(ds_holdout), len(trained_models)))

    with torch.no_grad():
        for i, model in enumerate(trained_models):
            preds_list = []
            for imgs, angs, _, _ in dl_holdout:
                imgs = imgs.to(Config.DEVICE)
                angs = angs.to(Config.DEVICE)

                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds_list.append(probs)

            ensemble_preds[:, i] = np.concatenate(preds_list).flatten()

    # Average predictions
    avg_preds = np.mean(ensemble_preds, axis=1)

    # Clip predictions to avoid log(0)
    avg_preds_clipped = np.clip(avg_preds, 1e-15, 1 - 1e-15)

    # Calculate Metric
    final_log_loss = log_loss(y_holdout, avg_preds_clipped)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")
    errors = np.abs(y_holdout - avg_preds)

    # Compute features for correlation analysis
    # X_holdout shape: (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_mean = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_holdout[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_holdout,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Calculate correlation
    corr = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Failure Analysis - Correlation with Error:")
    print(corr)

    # 6. Submission
    threshold = 0.14772333549413377
    if final_log_loss < threshold:
        logger.info(
            f"Validation metric {final_log_loss} is better than threshold {threshold}. Generating submission..."
        )

        # Create Test Dataset
        ds_test = IcebergDataset(
            test_data_full["images"],
            test_data_full["angles"],
            ids=test_data_full["ids"],
            transform=False,
            global_stats=global_stats,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Inference
        test_ensemble_preds = np.zeros((len(ds_test), len(trained_models)))

        with torch.no_grad():
            for i, model in enumerate(trained_models):
                preds_list = []
                for imgs, angs, _ in dl_test:
                    imgs = imgs.to(Config.DEVICE)
                    angs = angs.to(Config.DEVICE)

                    outputs = model(imgs, angs)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds_list.append(probs)

                test_ensemble_preds[:, i] = np.concatenate(preds_list).flatten()

        # Average
        test_avg_preds = np.mean(test_ensemble_preds, axis=1)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"id": test_data_full["ids"], "is_iceberg": test_avg_preds}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        logger.info(
            f"Validation metric {final_log_loss} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
