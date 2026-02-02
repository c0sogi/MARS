import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data_loader import load_and_process_data, IcebergDataset
from library.model import SimpleCNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, angles, labels in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # DSN_CNN expects (x, angle)
        logits = model(inputs, angles)

        # Calculate loss
        loss = criterion(logits, labels.view(-1, 1))

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average Log Loss.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, angles, labels in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            logits = model(inputs, angles)
            loss = criterion(logits, labels.view(-1, 1))

            running_loss += loss.item() * inputs.size(0)

    val_loss = running_loss / len(loader.dataset)
    return val_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns probabilities (sigmoid applied).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            angles = batch[1].to(device)

            logits = model(inputs, angles)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds).flatten()


def run_kfold_training():
    """
    Orchestrates the 5-Fold Cross-Validation training pipeline.
    """
    # Setup
    seed_everything(Config.SEED)
    logger = setup_logger("trainer", os.path.join(Config.WORKING_DIR, "train.log"))
    device = torch.device(Config.DEVICE)

    logger.info("Starting K-Fold Training...")

    # Load Data
    # We load cached data if available, otherwise process from scratch
    (train_data, val_data, test_data) = load_and_process_data(load_cached_data=True)

    X_train_part, y_train_part, angle_train_part = train_data
    X_val_part, y_val_part, angle_val_part = val_data
    X_test, ids_test, angle_test = test_data

    # Combine train and val for K-Fold splitting
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    angle_full = np.concatenate([angle_train_part, angle_val_part], axis=0)

    # Debug Mode: Truncate data
    if Config.DEBUG:
        logger.info(
            f"Debug mode enabled. Truncating to {Config.DEBUG_SAMPLES} samples."
        )
        limit = Config.DEBUG_SAMPLES
        X_full = X_full[:limit]
        y_full = y_full[:limit]
        angle_full = angle_full[:limit]
        X_test = X_test[:limit]
        ids_test = ids_test[:limit]
        angle_test = angle_test[:limit]

    # Prepare Test Loader (Common for all folds)
    test_dataset = IcebergDataset(X_test, angle_test, y=None, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # K-Fold Initialization
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Array to store accumulated test predictions
    test_preds_accum = np.zeros(len(X_test))

    # Define Transforms (Augmentation for training)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split data
        X_tr, y_tr, ang_tr = X_full[train_idx], y_full[train_idx], angle_full[train_idx]
        X_va, y_va, ang_va = X_full[val_idx], y_full[val_idx], angle_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

        # Create Loaders
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

        # Initialize Model
        model = SimpleCNN().to(device)

        # Optimizer and Loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            logger.info(
                f"Fold {fold+1} Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model for inference
        logger.info(
            f"Loading best weights for Fold {fold+1} with Val Loss: {best_val_loss}"
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Predict on Test Set
        fold_preds = predict(model, test_loader, device)
        test_preds_accum += fold_preds

        # Clean up to save memory
        del model, optimizer, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # Average predictions
    avg_preds = test_preds_accum / Config.N_FOLDS

    # Save Submission
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
