import os
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import (
    WORK_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEVICE,
)
from library.utils import seed_everything, get_logger
from library.roi_processing import generate_roi_cache
from library.data_loader import BraTSDataset, get_transforms
from library.trainer import Trainer
from library.network import VRAWIVModel

# Initialize Logger
logger = get_logger("runfile")


def main():
    # 1. Setup
    seed_everything(SEED)
    logger.info("Starting Runfile Execution...")

    # 2. Load Dataframes (using cache)
    # df_train_cv: Data from train_metadata.csv (used for 5-Fold CV)
    # df_val_holdout: Data from val_metadata.csv (used for final metric calculation)
    # df_test: Data from test_metadata.csv (used for submission)
    # Note: generate_roi_cache returns (train, val, test) based on metadata files
    df_train_cv, df_val_holdout, df_test = generate_roi_cache(load_cached_data=True)

    # 3. Prepare 5-Fold CV
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Store model paths for ensemble
    fold_model_paths = []

    # Training Loop
    # We loop through folds on the df_train_cv dataset
    X = df_train_cv.index.values
    y = df_train_cv["MGMT_value"].values

    logger.info(
        f"Starting {n_folds}-Fold Cross-Validation on {len(df_train_cv)} samples..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Dataframes
        df_fold_train = df_train_cv.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_cv.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = BraTSDataset(
            df_fold_train, transform=get_transforms("train"), is_train=True
        )
        val_dataset = BraTSDataset(
            df_fold_val, transform=get_transforms("val"), is_train=False
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Trainer (Creates fresh model)
        trainer = Trainer()

        # Train
        trainer.fit(train_loader, val_loader)

        # Rename and save best model for this fold
        src_path = os.path.join(WORK_DIR, "best_model.pth")
        dst_path = os.path.join(WORK_DIR, f"best_model_fold{fold}.pth")

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            fold_model_paths.append(dst_path)
            logger.info(f"Saved fold {fold} model to {dst_path}")
        else:
            logger.warning(f"No best model found for fold {fold}!")

    # 4. Final Validation on Hold-out Set (Ensemble)
    logger.info("\nRunning Ensemble Validation on Hold-out Set...")

    # Create Hold-out Loader
    val_holdout_dataset = BraTSDataset(
        df_val_holdout, transform=get_transforms("val"), is_train=False
    )
    val_holdout_loader = DataLoader(
        val_holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    holdout_preds = np.zeros(len(df_val_holdout))

    # We need a model instance to load weights into
    inference_model = VRAWIVModel().to(DEVICE)
    inference_model.eval()

    with torch.no_grad():
        for model_path in fold_model_paths:
            logger.info(f"Inferencing with {os.path.basename(model_path)}...")
            inference_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            fold_preds = []
            for images, _ in val_holdout_loader:
                images = images.to(DEVICE)
                outputs = inference_model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

            holdout_preds += np.array(fold_preds)

    # Average predictions
    if len(fold_model_paths) > 0:
        holdout_preds /= len(fold_model_paths)

    # Calculate Metric
    holdout_targets = df_val_holdout["MGMT_value"].values
    final_auc = roc_auc_score(holdout_targets, holdout_preds)

    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    logger.info("\nPerforming Failure Analysis...")
    errors = np.abs(holdout_targets - holdout_preds)

    # Correlation with Target (Class Imbalance/Difficulty)
    corr_target = np.corrcoef(errors, holdout_targets)[0, 1]
    print(f"Failure Analysis - Correlation with Target: {corr_target:.4f}")

    # Correlation with ID (Temporal/Site bias proxy)
    ids = df_val_holdout["BraTS21ID"].values
    corr_id = np.corrcoef(errors, ids)[0, 1]
    print(f"Failure Analysis - Correlation with Subject ID: {corr_id:.4f}")

    # 6. Submission
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        logger.info(
            f"\nValidation Metric ({final_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Create Test Loader
        test_dataset = BraTSDataset(
            df_test, transform=get_transforms("val"), is_train=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = np.zeros(len(df_test))

        with torch.no_grad():
            for model_path in fold_model_paths:
                inference_model.load_state_dict(
                    torch.load(model_path, map_location=DEVICE)
                )

                fold_preds = []
                for images in test_loader:
                    images = images.to(DEVICE)
                    outputs = inference_model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

                test_preds += np.array(fold_preds)

        if len(fold_model_paths) > 0:
            test_preds /= len(fold_model_paths)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": test_preds}
        )

        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {SUBMISSION_PATH}")

    else:
        logger.info(
            f"\nValidation Metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
