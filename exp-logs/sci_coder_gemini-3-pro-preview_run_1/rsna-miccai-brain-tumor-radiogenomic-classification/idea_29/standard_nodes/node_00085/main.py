"""
Stochastic-Depth Weight-Inflated Volumetric (SD-WIV) Network
Entry Point: runfile.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint, load_checkpoint
from library.data import load_data, MGMTDataset, get_transforms
from library.model import SDWIVNet
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    logger = get_logger("main")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # Load Train (for CV) and Val (Hold-out) and Test (Submission)
    # Using load_cached_data=True to leverage any existing preprocessed arrays
    logger.info("Loading datasets...")
    train_ids, train_images, train_labels = load_data("train", load_cached_data=True)
    val_ids, val_images, val_labels = load_data("val", load_cached_data=True)
    test_ids, test_images, _ = load_data("test", load_cached_data=True)

    logger.info(f"Train shape: {train_images.shape}")
    logger.info(f"Val shape: {val_images.shape}")
    logger.info(f"Test shape: {test_images.shape}")

    # 3. 5-Fold Cross-Validation on 'Train' set
    # We train 5 models on the 'train' metadata split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    trained_models = []
    fold_aucs = []

    # Loop over folds
    for fold, (train_idx, dev_idx) in enumerate(skf.split(train_images, train_labels)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Prepare Fold Data
        X_fold_train = train_images[train_idx]
        y_fold_train = train_labels[train_idx]
        X_fold_dev = train_images[dev_idx]
        y_fold_dev = train_labels[dev_idx]

        # Datasets
        train_dataset = MGMTDataset(
            X_fold_train, y_fold_train, transform=get_transforms("train"), mode="train"
        )
        dev_dataset = MGMTDataset(
            X_fold_dev, y_fold_dev, transform=get_transforms("val"), mode="val"
        )

        # Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Initialization
        model = SDWIVNet().to(device)

        # Optimizer & Loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_fold_auc = 0.0
        best_model_filename = f"best_model_fold{fold}.pth"

        for epoch in range(Config.EPOCHS):
            avg_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, device, epoch, logger
            )
            val_loss, val_auc = validate(dev_loader, model, criterion, device, logger)

            # Save Best
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                save_checkpoint(
                    model.state_dict(),
                    is_best=True,
                    filename=f"checkpoint_fold{fold}.pth",
                    best_filename=best_model_filename,
                )

        logger.info(f"Fold {fold+1} Best AUC: {best_fold_auc}")
        fold_aucs.append(best_fold_auc)

        # Load best model for ensemble
        best_model_path = os.path.join(Config.WORKING_DIR, best_model_filename)
        loaded_model = SDWIVNet().to(device)
        load_checkpoint(best_model_path, loaded_model, device=device)
        loaded_model.eval()
        trained_models.append(loaded_model)

    logger.info(f"\nAverage CV AUC: {np.mean(fold_aucs)}")

    # 4. Hold-out Validation (Ensemble Inference)
    logger.info("\n=== Hold-out Validation Assessment ===")

    holdout_dataset = MGMTDataset(
        val_images, val_labels, transform=get_transforms("val"), mode="val"
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Prediction
    holdout_preds = np.zeros(len(val_labels))

    with torch.no_grad():
        for i, model in enumerate(trained_models):
            model_preds = []
            for images, _ in holdout_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                model_preds.extend(probs)

            holdout_preds += np.array(model_preds)

    holdout_preds /= len(trained_models)

    # Compute Metric
    final_metric = roc_auc_score(val_labels, holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("\n=== Failure Analysis ===")

    # Calculate errors
    errors = np.abs(val_labels - holdout_preds)

    # Extract simple input features from the image tensors for correlation
    # Feature 1: Mean Intensity of the volume
    # Feature 2: Standard Deviation of the volume
    # Feature 3: Target Class

    # Compute stats on CPU numpy arrays
    # val_images shape: (N, H, W, 9)
    mean_intensities = np.mean(val_images, axis=(1, 2, 3))
    std_intensities = np.std(val_images, axis=(1, 2, 3))

    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Mean_Intensity": mean_intensities,
            "Std_Intensity": std_intensities,
            "Target": val_labels,
        }
    )

    # Correlations
    corr_mean = analysis_df["Error"].corr(analysis_df["Mean_Intensity"])
    corr_std = analysis_df["Error"].corr(analysis_df["Std_Intensity"])
    corr_target = analysis_df["Error"].corr(analysis_df["Target"])

    print(f"Correlation (Error vs Mean Intensity): {corr_mean}")
    print(f"Correlation (Error vs Std Intensity): {corr_std}")
    print(f"Correlation (Error vs Target Class): {corr_target}")

    # 6. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        logger.info(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_dataset = MGMTDataset(
            test_images, None, transform=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = np.zeros(len(test_images))

        with torch.no_grad():
            for model in trained_models:
                model_preds = []
                for images, _ in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    model_preds.extend(probs)
                test_preds += np.array(model_preds)

        test_preds /= len(trained_models)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
