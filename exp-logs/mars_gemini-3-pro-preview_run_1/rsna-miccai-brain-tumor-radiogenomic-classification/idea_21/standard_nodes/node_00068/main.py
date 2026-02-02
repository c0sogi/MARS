import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.train as train_lib


def main():
    # 1. Configuration & Setup
    # Override epochs for fast baseline execution as per requirements
    config.NUM_EPOCHS = 10

    # Ensure reproducibility
    utils.set_seed(config.SEED)
    logger = utils.get_logger(__name__)
    device = config.DEVICE

    logger.info(f"Initializing SF-WIV Pipeline on {device}...")

    # 2. Data Loading
    # Load all datasets using the provided processing function which handles caching
    logger.info("Loading datasets...")
    # Train data for CV
    train_imgs, train_lbls, train_ids = data.process_dataset(
        config.TRAIN_METADATA, "train", load_cached_data=True
    )
    # Fixed validation data for final evaluation
    val_imgs, val_lbls, val_ids = data.process_dataset(
        config.VAL_METADATA, "val", load_cached_data=True
    )
    # Test data for submission
    test_imgs, _, test_ids = data.process_dataset(
        config.TEST_METADATA, "test", load_cached_data=True
    )

    # 3. 5-Fold Cross-Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    fold_models = []

    logger.info(f"Starting {n_folds}-Fold Cross-Validation...")

    # Iterate through folds
    for fold, (train_idx, inner_val_idx) in enumerate(
        skf.split(train_imgs, train_lbls)
    ):
        logger.info(f"--- Fold {fold + 1}/{n_folds} ---")

        # Prepare Fold Data
        X_train_fold, y_train_fold = train_imgs[train_idx], train_lbls[train_idx]
        X_inner_val, y_inner_val = train_imgs[inner_val_idx], train_lbls[inner_val_idx]

        # Create Datasets
        train_ds = data.SFWIVDataset(
            X_train_fold, y_train_fold, transform=data.get_transforms("train")
        )
        inner_val_ds = data.SFWIVDataset(
            X_inner_val, y_inner_val, transform=data.get_transforms("val")
        )

        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        inner_val_loader = torch.utils.data.DataLoader(
            inner_val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = model_lib.SFWIVModel(pretrained=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Training Loop for this fold
        best_fold_auc = 0.0
        best_model_state = None

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_lib.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            # Validate on inner fold split to track progress
            val_loss, val_auc = train_lib.validate(
                model, inner_val_loader, criterion, device
            )

            # Save best state
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_model_state = model.state_dict().copy()

        logger.info(f"Fold {fold + 1} Best AUC: {best_fold_auc:.4f}")

        # Restore best model and store for ensemble
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        fold_models.append(model)

        # Save to disk as artifact
        torch.save(
            model.state_dict(),
            os.path.join(config.WORKING_DIR, f"best_model_fold{fold}.pth"),
        )

    # 4. Final Validation on Hold-Out Set
    logger.info("Running Final Validation on Hold-Out Set...")

    val_ds = data.SFWIVDataset(val_imgs, val_lbls, transform=data.get_transforms("val"))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Ensemble Inference
    val_probs_ensemble = []
    val_targets_all = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Get predictions from all fold models
            batch_probs = []
            for model in fold_models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average predictions
            avg_probs = np.mean(batch_probs, axis=0)
            val_probs_ensemble.append(avg_probs)
            val_targets_all.append(targets.numpy())

    val_probs_ensemble = np.concatenate(val_probs_ensemble).flatten()
    val_targets_all = np.concatenate(val_targets_all).flatten()

    # Calculate Metric
    final_auc = roc_auc_score(val_targets_all, val_probs_ensemble)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(val_targets_all - val_probs_ensemble)

    # Correlation with Target (Class Imbalance/Difficulty)
    corr_target, _ = pearsonr(errors, val_targets_all)
    print(f"Correlation between Error and Target Class: {corr_target:.4f}")

    # Correlation with Subject ID (Acquisition/Temporal shift)
    # Ensure IDs are numeric
    val_ids_numeric = val_ids.astype(float)
    corr_id, _ = pearsonr(errors, val_ids_numeric)
    print(f"Correlation between Error and Subject ID: {corr_id:.4f}")

    # 6. Submission
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({final_auc:.4f}) > Threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        test_ds = data.SFWIVDataset(
            test_imgs, labels=None, transform=data.get_transforms("test")
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        test_probs_ensemble = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                batch_probs = []
                for model in fold_models:
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    batch_probs.append(probs.cpu().numpy())

                avg_probs = np.mean(batch_probs, axis=0)
                test_probs_ensemble.append(avg_probs)

        test_probs_ensemble = np.concatenate(test_probs_ensemble).flatten()

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_ids, "MGMT_value": test_probs_ensemble}
        )

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation AUC ({final_auc:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
