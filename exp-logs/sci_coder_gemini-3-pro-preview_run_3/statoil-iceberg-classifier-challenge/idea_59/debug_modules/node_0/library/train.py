import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, get_logger
from library.model import ACICNN
from library.data_loader import get_data, get_fold_loaders, get_test_loader

logger = get_logger("train")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for imgs, raw_angs, norm_angs, labels in loader:
        imgs = imgs.to(device)
        raw_angs = raw_angs.to(device)
        norm_angs = norm_angs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (x, raw_angle, norm_angle)
        outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        dataset_size += imgs.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, predictions_array, targets_array)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    with torch.no_grad():
        for imgs, raw_angs, norm_angs, labels in loader:
            imgs = imgs.to(device)
            raw_angs = raw_angs.to(device)
            norm_angs = norm_angs.to(device)
            labels = labels.to(device)

            outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            dataset_size += imgs.size(0)

            # Apply sigmoid for probability predictions
            batch_preds = torch.sigmoid(outputs).cpu().numpy()
            preds.append(batch_preds)
            targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    return avg_loss, np.concatenate(preds), np.concatenate(targets)


def run_fold(fold_idx, data):
    """
    Runs training and validation for a single fold.

    Args:
        fold_idx (int): The fold index (0-4).
        data (dict): The data dictionary containing X, y, etc.

    Returns:
        tuple: (best_val_loss, scaler, imputation_val, checkpoint_path)
    """
    logger.info(f"Starting Fold {fold_idx}")

    # Get leak-free loaders and preprocessing stats
    train_loader, val_loader, scaler, imp_val = get_fold_loaders(
        fold_idx, data, batch_size=Config.BATCH_SIZE
    )

    # Initialize Model
    model = ACICNN().to(Config.DEVICE)

    # Optimizer and Loss
    # Using constant learning rate as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, _, _ = validate(model, val_loader, criterion, Config.DEVICE)

        # Print full precision metrics
        logger.info(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_loss, scaler, imp_val, checkpoint_path


def train_model():
    """
    Main function to execute the 5-fold cross-validation training pipeline
    and generate the submission file.
    """
    seed_everything(Config.SEED)
    Config.setup()

    # Load Data
    data = get_data(load_cached_data=True)

    # Prepare storage for predictions
    # OOF predictions for validation scoring
    oof_preds = np.zeros(len(data["y_train"]))
    # Test predictions accumulator
    test_preds_accum = np.zeros(len(data["X_test"]))

    # Stratified K-Fold logic is handled inside get_fold_loaders via indices,
    # but we need the indices here to map OOF predictions back to the array.
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(data["X_train"], data["y_train"]))

    for fold_idx in range(Config.NUM_FOLDS):
        # 1. Train the fold
        best_loss, scaler, imp_val, ckpt_path = run_fold(fold_idx, data)

        # 2. Load best model for inference
        model = ACICNN().to(Config.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model.eval()
        criterion = nn.BCEWithLogitsLoss()

        # 3. Generate OOF Predictions
        # We need to recreate the val loader to ensure correct order
        _, val_loader, _, _ = get_fold_loaders(
            fold_idx, data, batch_size=Config.BATCH_SIZE
        )
        _, val_preds, _ = validate(model, val_loader, criterion, Config.DEVICE)

        # Map predictions to correct indices
        _, val_idx = splits[fold_idx]
        oof_preds[val_idx] = val_preds

        # 4. Generate Test Predictions
        # Use the specific scaler/imputer from this fold to prevent leakage/mismatch
        test_loader = get_test_loader(
            data, scaler, imp_val, batch_size=Config.BATCH_SIZE
        )

        fold_test_preds = []
        with torch.no_grad():
            for imgs, raw_angs, norm_angs in test_loader:
                imgs = imgs.to(Config.DEVICE)
                raw_angs = raw_angs.to(Config.DEVICE)
                norm_angs = norm_angs.to(Config.DEVICE)

                outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
                fold_test_preds.append(torch.sigmoid(outputs).cpu().numpy())

        test_preds_accum += np.concatenate(fold_test_preds) / Config.NUM_FOLDS

        logger.info(f"Fold {fold_idx} completed. Best Val Loss: {best_loss}")

    # Calculate and print overall OOF score
    overall_score = log_loss(data["y_train"], oof_preds)
    logger.info(f"Overall OOF Log Loss: {overall_score}")

    # Save Submission
    submission = pd.DataFrame({"id": data["ids_test"], "is_iceberg": test_preds_accum})

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
