import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, set_seed
from library.utils import get_logger, EarlyStopping
from library.data_loader import get_dataloaders
from library.model import SIWBN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_fold(fold_index, train_data, logger=None):
    """
    Runs the training and validation for a single fold.

    Args:
        fold_index (int): The index of the current fold (0-4).
        train_data (tuple): Tuple containing (X, inc, y, ids) for the full training set.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        model (nn.Module): The model loaded with the best weights from this fold.
        best_loss (float): The best validation loss achieved.
    """
    # Ensure reproducibility for this fold
    set_seed(Config.SEED + fold_index)

    if logger is None:
        logger = get_logger(os.path.join(Config.WORKING_DIR, "train.log"))

    logger.info(f"Starting Fold {fold_index + 1}/{Config.NUM_FOLDS}")

    # 1. Data Loaders
    train_loader, val_loader = get_dataloaders(fold_index, train_data)

    # 2. Model Initialization
    device = torch.device(Config.DEVICE)
    model = SIWBN().to(device)

    # 3. Optimization Setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 4. Early Stopping
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_index}.pth")
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        verbose=True,
        path=checkpoint_path,
        trace_func=logger.info,
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging (Full precision as requested)
        logger.info(
            f"Fold {fold_index} Epoch {epoch + 1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Early Stopping check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # 6. Load Best Weights
    logger.info(f"Loading best weights for Fold {fold_index} from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # Return best score from early stopping tracker
    return model, -early_stopping.best_score
