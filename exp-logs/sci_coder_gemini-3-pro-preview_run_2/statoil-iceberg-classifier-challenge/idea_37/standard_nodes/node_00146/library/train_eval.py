import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import (
    DEVICE,
    CACHE_DIR,
    SUBMISSION_DIR,
    RANDOM_SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    NUM_FOLDS,
)
from library.utils import get_logger, set_seed
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import RDPWBN

logger = get_logger("train_eval")


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state to disk and keeps a copy in memory.
    """

    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            # Save the best model to disk
            torch.save(self.best_model_state, self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, angles, labels in dataloader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, angles, labels in dataloader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = correct / total

    # Print full precision metrics
    logger.info(f"Validation Loss: {val_loss}")
    logger.info(f"Validation Accuracy: {val_acc}")

    return val_loss, val_acc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            angles = batch[1].to(device)

            outputs = model(inputs, angles)
            probs = torch.sigmoid(outputs)
            preds_list.extend(probs.cpu().numpy().flatten())

    return np.array(preds_list)


def train_fold(
    fold_idx, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE, device=DEVICE
):
    """
    Trains a model for a single fold using Early Stopping and Scheduler.
    """
    logger.info(f"Starting training for Fold {fold_idx}")

    model = RDPWBN().to(device)

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion = nn.BCEWithLogitsLoss()

    # Path to save the best model for this fold
    save_path = os.path.join(CACHE_DIR, f"model_fold_{fold_idx}.pth")
    early_stopping = EarlyStopping(patience=patience, path=save_path)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} Train Acc: {train_acc}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model weights before returning
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    else:
        logger.warning(
            f"Fold {fold_idx}: No best model state found. Using final state."
        )
        torch.save(model.state_dict(), save_path)

    return model


def run_kfold_and_submission(epochs=EPOCHS, patience=PATIENCE, load_cached_data=True):
    """
    Executes the full Stratified K-Fold training pipeline and generates the submission file.
    """
    set_seed(RANDOM_SEED)

    # 1. Load Processed Data
    data = process_and_cache_data(load_cached_data=load_cached_data)

    X = data["train_images"]
    angles = data["train_angles"]
    y = data["train_labels"]

    X_test = data["test_images"]
    angles_test = data["test_angles"]
    test_ids = data["test_ids"]

    # 2. Prepare Test Loader
    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Create Datasets
        # Train: Transform=True (Augmentation)
        train_ds = IcebergDataset(
            X[train_idx], angles[train_idx], y[train_idx], transform=True
        )
        # Val: Transform=False
        val_ds = IcebergDataset(
            X[val_idx], angles[val_idx], y[val_idx], transform=False
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Train Model for this fold
        model = train_fold(
            fold, train_loader, val_loader, epochs=epochs, patience=patience
        )

        # Predict on Test Set
        preds = predict(model, test_loader, DEVICE)
        fold_preds.append(preds)

    # 4. Aggregate Predictions (Mean)
    avg_preds = np.mean(fold_preds, axis=0)

    # 5. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
    df_sub.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
