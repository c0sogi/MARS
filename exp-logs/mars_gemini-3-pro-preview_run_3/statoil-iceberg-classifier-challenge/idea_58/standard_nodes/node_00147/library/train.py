import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import (
    DEVICE,
    NUM_FOLDS,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    NUM_WORKERS,
    SEED,
    IDEA_ID,
)
from library.utils import set_seed, setup_logger, save_checkpoint, AverageMeter
from library.data import IcebergDataset, load_and_process_json, get_transforms
from library.model import CDICNN


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, angles, targets) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).float()

        # Forward pass
        outputs = model(images, angles)

        # Ensure targets match output shape (B,)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    logger.info(f"Epoch [{epoch}] Train Loss: {losses.avg:.6f}")
    return losses.avg


def validate(val_loader, model, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).float()

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    # Calculate Log Loss
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    # Clip predictions to avoid log(0) error, though sklearn handles this usually
    final_log_loss = log_loss(y_true, y_pred)

    logger.info(f"Validation Log Loss: {final_log_loss:.15f}")
    return final_log_loss


def inference(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)

    return np.array(all_preds)


def run_training(debug_sample_size=None):
    """
    Main training loop implementing 5-Fold Cross-Validation.
    """
    set_seed(SEED)
    logger = setup_logger(os.path.join(CHECKPOINT_DIR, "train.log"))
    logger.info(f"Starting training for {IDEA_ID} with {NUM_FOLDS}-Fold CV")

    # 1. Load Data
    # Load full training data
    logger.info("Loading training data...")
    X_full, ang_full, y_full, ids_train = load_and_process_json(
        TRAIN_JSON_PATH, "train_full", load_cached=True
    )

    # Load test data
    logger.info("Loading test data...")
    X_test, ang_test, _, ids_test = load_and_process_json(
        TEST_JSON_PATH, "test", load_cached=True
    )

    # Debug mode
    if debug_sample_size:
        logger.info(f"Debug mode: limiting data to {debug_sample_size} samples")
        X_full = X_full[:debug_sample_size]
        ang_full = ang_full[:debug_sample_size]
        y_full = y_full[:debug_sample_size]
        X_test = X_test[:debug_sample_size]
        ang_test = ang_test[:debug_sample_size]
        ids_test = ids_test[:debug_sample_size]

    # 2. Prepare Test Data (Imputation)
    # Impute missing angles in Test set using global median of training data
    valid_train_angles = ang_full[~np.isnan(ang_full)]
    global_median = (
        np.median(valid_train_angles) if len(valid_train_angles) > 0 else 0.0
    )
    ang_test[np.isnan(ang_test)] = global_median

    test_dataset = IcebergDataset(
        X_test, ang_test, None, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Array to store predictions from each fold
    fold_test_preds = np.zeros((len(X_test), NUM_FOLDS))

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n========== Fold {fold} ==========")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        ang_train_fold, ang_val_fold = ang_full[train_idx], ang_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Impute Angles (Compute median on training fold only to prevent leakage)
        valid_fold_angles = ang_train_fold[~np.isnan(ang_train_fold)]
        fold_median = (
            np.median(valid_fold_angles) if len(valid_fold_angles) > 0 else 0.0
        )

        ang_train_fold[np.isnan(ang_train_fold)] = fold_median
        ang_val_fold[np.isnan(ang_val_fold)] = fold_median

        # Create Datasets and Loaders
        train_ds = IcebergDataset(
            X_train_fold,
            ang_train_fold,
            y_train_fold,
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, transform=get_transforms("val")
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Criterion
        model = CDICNN().to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0

        # Training Loop
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, DEVICE, epoch, logger
            )
            val_loss = validate(val_loader, model, criterion, DEVICE, logger)

            # Early Stopping & Checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    is_best=True,
                    checkpoint_dir=CHECKPOINT_DIR,
                    fold_idx=fold,
                )
                logger.info(
                    f"New best model found at epoch {epoch} with loss {val_loss:.6f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Load best model for inference
        logger.info(f"Loading best model for Fold {fold}...")
        best_model_path = os.path.join(CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Inference on Test Set
        logger.info(f"Generating predictions for Fold {fold}...")
        preds = inference(test_loader, model, DEVICE)
        fold_test_preds[:, fold] = preds

    # 4. Ensembling and Submission
    logger.info("\n========== Ensembling ==========")
    avg_preds = np.mean(fold_test_preds, axis=1)

    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
    logger.info("Training complete.")
