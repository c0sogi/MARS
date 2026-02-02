import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from sklearn.model_selection import KFold
import logging
import time

# Import library modules
from library.utils import set_seed, setup_logger, save_checkpoint
from library.data import get_datasets, IcebergDataset, get_transforms
from library.model import IcebergResNet18GeM
from library.training import Trainer
from library.inference import predict_tta, select_pseudo_labels

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
CV_FOLDS = 5
MAX_EPOCHS_CV = 30  # Cap for speed
SWA_EPOCHS = 10
SUBMISSION_THRESHOLD = 0.16918645240183008
OUTPUT_DIR = "./working/output"
SUBMISSION_DIR = "./submission"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Setup Logger
logger = setup_logger("main", os.path.join(OUTPUT_DIR, "training.log"))


def main():
    set_seed(SEED)
    logger.info(f"Using device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading datasets...")
    # train_ds is the labeled training set (80% of original)
    # val_ds is the hold-out validation set (20% of original)
    # test_ds is the unlabeled test set
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True)

    # We use the 'train_ds' for Cross-Validation and Teacher Training
    # 'val_ds' is strictly reserved for the final metric reporting

    # -------------------------------------------------------------------------
    # 2. Stage 1: Teacher Ensemble - Cross Validation for Optimal Epochs
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 1: Finding Optimal Convergence Epoch via CV...")

    # Create K-Fold splits on train_ds
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    indices = np.arange(len(train_ds))

    best_epochs = []

    # We need to access the underlying data to subset correctly
    # IcebergDataset doesn't support direct indexing for subsetting in a way that preserves attributes easily
    # So we use torch.utils.data.Subset

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(indices)):
        logger.info(f"CV Fold {fold+1}/{CV_FOLDS}")

        # Create Subsets
        fold_train_ds = Subset(train_ds, train_idx)
        fold_val_ds = Subset(train_ds, valid_idx)

        # Loaders
        fold_train_loader = DataLoader(
            fold_train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        fold_val_loader = DataLoader(
            fold_val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        model = IcebergResNet18GeM(pretrained=True)

        # Trainer
        fold_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"fold_{fold}")
        trainer = Trainer(model, DEVICE, logger=logger)

        # Fit (Standard Training with Early Stopping)
        # We use a slightly lower patience here to find the "elbow" quickly
        trainer.fit(
            fold_train_loader,
            fold_val_loader,
            epochs=MAX_EPOCHS_CV,
            checkpoint_dir=fold_checkpoint_dir,
        )

        # Load best checkpoint to find the epoch
        best_ckpt = torch.load(
            os.path.join(fold_checkpoint_dir, "best_model.pth"), map_location=DEVICE
        )
        best_epochs.append(best_ckpt["epoch"])

        # Cleanup to save memory
        del model, trainer, fold_train_loader, fold_val_loader
        torch.cuda.empty_cache()

    avg_best_epoch = int(np.mean(best_epochs))
    logger.info(f"Optimal Convergence Epoch (Average): {avg_best_epoch}")

    # -------------------------------------------------------------------------
    # 3. Stage 1: Teacher Ensemble - Production Training
    # -------------------------------------------------------------------------
    logger.info("Training Teacher Ensemble on full labeled training set...")

    teacher_models = []
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    # We use the hold-out val set here just for monitoring, not for model selection (we use fixed epochs)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Train 5 independent teachers
    for i in range(5):
        logger.info(f"Training Teacher Model {i+1}/5")
        set_seed(SEED + i)  # Different seed for diversity

        model = IcebergResNet18GeM(pretrained=True)
        trainer = Trainer(model, DEVICE, logger=logger)

        model_ckpt_dir = os.path.join(CHECKPOINT_DIR, f"teacher_{i}")

        # Standard Phase (Fixed Epochs based on CV)
        # We pass the calculated epochs. Trainer.fit will still use ES if val loss degrades,
        # but we expect it to reach near avg_best_epoch.
        trainer.fit(
            train_loader,
            val_loader,
            epochs=avg_best_epoch + 2,
            checkpoint_dir=model_ckpt_dir,
        )

        # SWA Phase
        trainer.fit_swa(
            train_loader,
            val_loader,
            swa_epochs=SWA_EPOCHS,
            checkpoint_dir=model_ckpt_dir,
        )

        # Load final SWA model for inference
        swa_path = os.path.join(model_ckpt_dir, "swa_model.pth")
        swa_ckpt = torch.load(swa_path, map_location=DEVICE)
        model.load_state_dict(swa_ckpt["state_dict"])
        model.eval()
        teacher_models.append(model)

    # -------------------------------------------------------------------------
    # 4. Pseudo-Labeling
    # -------------------------------------------------------------------------
    logger.info("Generating Pseudo-Labels...")

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    ensemble_preds = []
    for model in teacher_models:
        preds = predict_tta(model, test_loader, DEVICE)
        ensemble_preds.append(preds)

    # Filter Pseudo-Labels
    pseudo_df = select_pseudo_labels(
        ensemble_preds, confidence_threshold=0.95, variance_threshold=0.02
    )

    # Create Pseudo-Label Dataset
    if len(pseudo_df) > 0:
        logger.info(f"Creating Pseudo-Label Dataset with {len(pseudo_df)} samples.")

        # Map IDs to indices in test_ds
        test_id_to_idx = {uid: idx for idx, uid in enumerate(test_ds.ids)}

        pseudo_indices = [test_id_to_idx[uid] for uid in pseudo_df["id"].values]
        pseudo_labels = pseudo_df["is_iceberg"].values.astype(np.float32)

        # Extract data
        pseudo_images = test_ds.images[pseudo_indices]
        pseudo_angles = test_ds.angles[pseudo_indices]
        pseudo_ids = test_ds.ids[pseudo_indices]

        # Create Dataset
        pseudo_ds = IcebergDataset(
            pseudo_images,
            pseudo_angles,
            pseudo_ids,
            pseudo_labels,
            transform=get_transforms("train"),  # Apply train transforms (augmentation)
            stats=train_ds.stats,
        )

        # Combine
        combined_ds = ConcatDataset([train_ds, pseudo_ds])
    else:
        logger.warning(
            "No pseudo-labels selected. Proceeding with original training data only."
        )
        combined_ds = train_ds

    # Clear memory
    del teacher_models
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Stage 2: Student Ensemble Training
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 2: Student Ensemble Training...")

    student_models = []
    combined_loader = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Train 5 independent students
    for i in range(5):
        logger.info(f"Training Student Model {i+1}/5")
        set_seed(SEED + 100 + i)

        model = IcebergResNet18GeM(pretrained=True)
        trainer = Trainer(model, DEVICE, logger=logger)

        model_ckpt_dir = os.path.join(CHECKPOINT_DIR, f"student_{i}")

        # Train
        trainer.fit(
            combined_loader,
            val_loader,
            epochs=avg_best_epoch + 2,
            checkpoint_dir=model_ckpt_dir,
        )
        trainer.fit_swa(
            combined_loader,
            val_loader,
            swa_epochs=SWA_EPOCHS,
            checkpoint_dir=model_ckpt_dir,
        )

        # Load SWA
        swa_path = os.path.join(model_ckpt_dir, "swa_model.pth")
        swa_ckpt = torch.load(swa_path, map_location=DEVICE)
        model.load_state_dict(swa_ckpt["state_dict"])
        model.eval()
        student_models.append(model)

    # -------------------------------------------------------------------------
    # 6. Final Evaluation
    # -------------------------------------------------------------------------
    logger.info("Evaluating Student Ensemble on Hold-Out Validation Set...")

    val_preds_list = []
    for model in student_models:
        preds = predict_tta(model, val_loader, DEVICE)
        val_preds_list.append(preds)

    # Average predictions
    final_val_preds = {}
    all_ids = list(val_preds_list[0].keys())

    y_true = []
    y_pred = []

    # Create a map for ground truth
    val_id_to_label = {uid: label for uid, label in zip(val_ds.ids, val_ds.labels)}

    for uid in all_ids:
        avg_prob = np.mean([p[uid] for p in val_preds_list])
        final_val_preds[uid] = avg_prob

        y_true.append(val_id_to_label[uid])
        y_pred.append(avg_prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip for Log Loss
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_loss = -np.mean(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    print(f"Final Validation Metric: {log_loss}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    errors = np.abs(y_true - y_pred)

    # Extract features from val_ds
    # val_ds.images is (N, 75, 75, 2)
    b1_means = np.mean(val_ds.images[..., 0], axis=(1, 2))
    b2_means = np.mean(val_ds.images[..., 1], axis=(1, 2))
    angles = val_ds.angles

    # Handle NaNs in angles for correlation (fill with mean)
    angles_filled = angles.copy()
    angles_filled[np.isnan(angles_filled)] = np.nanmean(angles_filled)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_filled,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    logger.info("Correlation of Error with Features:")
    logger.info(correlations)
    print("Failure Analysis Correlations:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 8. Submission
    # -------------------------------------------------------------------------
    if log_loss < SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation Metric ({log_loss}) is better than threshold ({SUBMISSION_THRESHOLD}). Generating Submission..."
        )

        test_preds_list = []
        for model in student_models:
            preds = predict_tta(model, test_loader, DEVICE)
            test_preds_list.append(preds)

        # Average
        final_submission = {}
        test_ids = list(test_preds_list[0].keys())

        for uid in test_ids:
            avg_prob = np.mean([p[uid] for p in test_preds_list])
            final_submission[uid] = avg_prob

        # Save
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub = pd.DataFrame(
            list(final_submission.items()), columns=["id", "is_iceberg"]
        )
        df_sub.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        logger.info(
            f"Validation Metric ({log_loss}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
