import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, seed_everything
from library.utils import get_logger
from library.train import run_fold
from library.dataset import get_dataset, IcebergDataset, get_transforms
from library.model import IcebergResNet


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("Runfile")

    logger.info("Starting 5-Fold Cross-Validation execution...")

    # 2. Prepare Data for CV
    # We use the full 'train' metadata set for CV training
    train_ds_full = get_dataset("train", load_cached_data=True)

    X = train_ds_full.images
    A = train_ds_full.angles
    y = train_ds_full.labels
    ids = train_ds_full.ids

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []

    # 3. Train Models (5 Folds)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- FOLD {fold} ---")

        # Create Fold Datasets
        # Note: We must apply correct transforms (augmentation for train, none for val)
        train_fold = IcebergDataset(
            X[train_idx],
            A[train_idx],
            y[train_idx],
            ids[train_idx],
            transform=get_transforms("train"),
        )
        val_fold = IcebergDataset(
            X[val_idx],
            A[val_idx],
            y[val_idx],
            ids[val_idx],
            transform=get_transforms("val"),
        )

        # Train
        model = run_fold(train_fold, val_fold, fold_idx=fold)
        models.append(model)

    # 4. Validation & Failure Analysis (Ensemble on Hold-out Set)
    logger.info("Performing ensemble validation on hold-out set...")
    device = torch.device(Config.DEVICE)

    # Load the fixed hold-out validation set
    holdout_dataset = get_dataset("val", load_cached_data=True)
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    # Inference on Hold-out Set using Ensemble
    with torch.no_grad():
        for images, angles, labels, _ in holdout_loader:
            images = images.to(device)
            angles = angles.to(device)

            batch_preds = []
            for model in models:
                model.eval()
                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs)

            # Average predictions across models
            avg_probs = torch.stack(batch_preds).mean(dim=0)

            all_preds.extend(avg_probs.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate Metric
    final_metric = log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Extract features for correlation from raw dataset arrays
    raw_images = holdout_dataset.images
    raw_angles = holdout_dataset.angles

    b1_means = np.mean(raw_images[:, :, :, 0], axis=(1, 2))
    b2_means = np.mean(raw_images[:, :, :, 1], axis=(1, 2))

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": raw_angles,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    df_analysis.fillna(0, inplace=True)

    logger.info("Failure Analysis - Correlation with Error Magnitude:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.21099163245555455
    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.6f} < {THRESHOLD}. Generating submission..."
        )

        test_dataset = get_dataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                batch_preds = []
                for model in models:
                    model.eval()

                    # TTA: Original
                    logits_orig = model(images, angles)
                    probs_orig = torch.sigmoid(logits_orig)

                    # TTA: Horizontal Flip
                    images_h = torch.flip(images, dims=[3])
                    logits_h = model(images_h, angles)
                    probs_h = torch.sigmoid(logits_h)

                    # TTA: Vertical Flip
                    images_v = torch.flip(images, dims=[2])
                    logits_v = model(images_v, angles)
                    probs_v = torch.sigmoid(logits_v)

                    # Average Probabilities for this model
                    probs_avg_model = (probs_orig + probs_h + probs_v) / 3.0
                    batch_preds.append(probs_avg_model)

                # Average predictions across all models in ensemble
                final_probs = torch.stack(batch_preds).mean(dim=0)

                test_preds.extend(final_probs.cpu().numpy().flatten())
                test_ids.extend(ids)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric {final_metric:.6f} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
