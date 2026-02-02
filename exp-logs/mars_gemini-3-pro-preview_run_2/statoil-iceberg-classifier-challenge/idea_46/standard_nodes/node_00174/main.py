import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config, set_seed, setup_directories
from library.utils import get_logger
from library.data_processing import load_and_process_data, IcebergDataset
from library.model import DualStreamWideBodyNetwork
from library.training import Trainer


def run():
    # 1. Setup
    set_seed(Config.SEED)
    setup_directories()
    logger = get_logger("RunFile")

    # 2. Load Data
    # load_cached_data=True to use pre-processed npz if available
    train_data, val_data, test_data, global_stats = load_and_process_data(
        load_cached_data=True
    )

    # We use the training data from train.csv for the 5-fold CV process
    X_train_full = train_data["images"]
    angle_train_full = train_data["angles"]
    y_train_full = train_data["labels"]

    # 3. Stratified K-Fold Training
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    model_paths = []

    logger.info(
        f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation on {len(X_train_full)} samples..."
    )

    for fold, (train_idx, inner_val_idx) in enumerate(
        skf.split(X_train_full, y_train_full)
    ):
        logger.info(f"\n=== Fold {fold} ===")

        # Split Data
        X_tr, X_iv = X_train_full[train_idx], X_train_full[inner_val_idx]
        a_tr, a_iv = angle_train_full[train_idx], angle_train_full[inner_val_idx]
        y_tr, y_iv = y_train_full[train_idx], y_train_full[inner_val_idx]

        # Datasets
        # transform=True for training to apply augmentation
        ds_train = IcebergDataset(X_tr, a_tr, y_tr, stats=global_stats, transform=True)
        ds_val = IcebergDataset(X_iv, a_iv, y_iv, stats=global_stats, transform=False)

        # Loaders
        dl_train = DataLoader(
            ds_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Trainer
        model = DualStreamWideBodyNetwork()
        trainer = Trainer(model, dl_train, dl_val, device=Config.DEVICE)

        # Fit
        save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        trainer.fit(save_path=save_path)
        model_paths.append(save_path)

        # Cleanup to save memory
        del model, trainer, dl_train, dl_val, ds_train, ds_val
        torch.cuda.empty_cache()

    # 4. Hold-out Validation (Ensemble)
    logger.info("\n=== Hold-out Validation ===")

    # Load Hold-out Data (val.csv)
    X_holdout = val_data["images"]
    a_holdout = val_data["angles"]
    y_holdout = val_data["labels"]

    ds_holdout = IcebergDataset(
        X_holdout, a_holdout, y_holdout, stats=global_stats, transform=False
    )
    dl_holdout = DataLoader(
        ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    ensemble_preds = np.zeros(len(y_holdout))

    for fold, path in enumerate(model_paths):
        logger.info(f"Evaluating model fold {fold}...")
        model = DualStreamWideBodyNetwork()
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in dl_holdout:
                images = images.to(Config.DEVICE)
                angles = angles.to(Config.DEVICE)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        ensemble_preds += np.array(fold_preds)

        del model
        torch.cuda.empty_cache()

    # Average Predictions
    ensemble_preds /= Config.NUM_FOLDS

    # Compute Metric
    # Clip predictions to avoid log(0)
    ensemble_preds_clipped = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_holdout, ensemble_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("\n=== Failure Analysis ===")

    # Calculate error magnitude
    errors = np.abs(y_holdout - ensemble_preds)

    # Extract features for correlation
    # 1. Incidence Angle
    # 2. Image Mean Intensity (Band 1 & 2)
    # Note: Images are (N, 75, 75, 3).

    img_means = np.mean(X_holdout, axis=(1, 2, 3))
    img_stds = np.std(X_holdout, axis=(1, 2, 3))

    # Correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": a_holdout,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.14772333549413377

    if final_metric < THRESHOLD:
        logger.info(
            f"\nValidation metric {final_metric:.6f} < {THRESHOLD}. Generating submission..."
        )

        X_test = test_data["images"]
        a_test = test_data["angles"]
        ids_test = test_data["ids"]

        ds_test = IcebergDataset(
            X_test, a_test, labels=None, stats=global_stats, transform=False
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_preds = np.zeros(len(ids_test))

        for fold, path in enumerate(model_paths):
            model = DualStreamWideBodyNetwork()
            model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles, _ in dl_test:
                    images = images.to(Config.DEVICE)
                    angles = angles.to(Config.DEVICE)
                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            test_ensemble_preds += np.array(fold_preds)
            del model
            torch.cuda.empty_cache()

        test_ensemble_preds /= Config.NUM_FOLDS

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_ensemble_preds})

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"\nValidation metric {final_metric:.6f} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
