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
from library.utils import get_logger, seed_everything
from library.data import IcebergDataset, _load_and_process_data
from library.model import DSICNN

# Initialize logger
logger = get_logger("train.py")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    val_loss = running_loss / len(loader.dataset)
    return val_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Test data loader.
        device (torch.device): Device to run on.

    Returns:
        np.ndarray: Flattened array of probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds).flatten()


def run_fold(fold_idx, train_loader, val_loader):
    """
    Runs the training and validation loop for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Loader for training subset.
        val_loader (DataLoader): Loader for validation subset.

    Returns:
        tuple: (path_to_best_model, best_validation_loss)
    """
    logger.info(f"Starting Fold {fold_idx}")

    device = Config.DEVICE
    model = DSICNN().to(device)

    # Optimizer: AdamW with constant learning rate and weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(
                    f"Fold {fold_idx} | Early stopping triggered at epoch {epoch+1}"
                )
                break

    logger.info(f"Fold {fold_idx} | Best Val Loss: {best_val_loss:.6f}")
    return best_model_path, best_val_loss


def main_cross_validation():
    """
    Orchestrates the 5-Fold Cross-Validation training and submission generation.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    # We use the internal library function to get raw arrays.
    # We merge the library's default train/val split to perform our own 5-Fold CV.
    data = _load_and_process_data(load_cached_data=True)

    X_total = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_total = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    angles_total = np.concatenate([data["meta_train"], data["meta_val"]], axis=0)

    X_test = data["X_test"]
    angles_test = data["meta_test"]

    # 2. Prepare Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    fold_model_paths = []
    cv_scores = []

    # 3. Training Loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_total, y_total)):
        # Create subsets
        X_train_fold, X_val_fold = X_total[train_idx], X_total[val_idx]
        y_train_fold, y_val_fold = y_total[train_idx], y_total[val_idx]
        ang_train_fold, ang_val_fold = angles_total[train_idx], angles_total[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold, ang_train_fold, y_train_fold, transform=train_transform
        )
        val_dataset = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, transform=None
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        # Run Fold
        model_path, best_loss = run_fold(fold_idx, train_loader, val_loader)
        fold_model_paths.append(model_path)
        cv_scores.append(best_loss)

    logger.info(f"CV Complete. Average Log Loss: {np.mean(cv_scores):.6f}")

    # 4. Inference and Submission
    logger.info("Generating predictions on test set...")

    test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_preds = np.zeros(len(X_test))

    for i, model_path in enumerate(fold_model_paths):
        logger.info(f"Predicting with model from Fold {i}...")
        model = DSICNN().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

        fold_preds = predict(model, test_loader, Config.DEVICE)
        test_preds += fold_preds

    # Average predictions (Ensembling)
    test_preds /= Config.NUM_FOLDS

    # Load test metadata to get IDs
    df_test = pd.read_csv(Config.TEST_META_PATH)

    submission = pd.DataFrame({"id": df_test["id"], "is_iceberg": test_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
