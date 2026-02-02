import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, AverageMeter, save_checkpoint, set_seed
from library.data_loader import get_kfold_loaders, get_test_loader
from library.model import MLCWNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Computation device (CPU/GPU).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Computation device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_fold(
    fold_idx, train_loader, val_loader, logger, num_epochs=Config.NUM_EPOCHS
):
    """
    Trains a single fold with Early Stopping and Scheduler.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        logger (Logger): Logger instance.
        num_epochs (int): Maximum number of epochs to train.
    """
    device = torch.device(Config.DEVICE)
    model = MLCWNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    logger.info(f"Starting training for Fold {fold_idx + 1}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Log metrics with full precision
        logger.info(
            f"Fold {fold_idx + 1} Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.16f} - Val Loss: {val_loss:.16f}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Deepcopy to ensure we save the exact weights that produced best_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save the best model for this fold
    if best_model_state is not None:
        save_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold_idx}.pth")
        save_checkpoint(best_model_state, save_path)
        logger.info(
            f"Saved best model for fold {fold_idx + 1} to {save_path} (Val Loss: {best_val_loss:.16f})"
        )
    else:
        logger.warning(f"No best model saved for fold {fold_idx + 1}!")


def train_model(num_epochs=Config.NUM_EPOCHS):
    """
    Orchestrates the Stratified K-Fold Cross-Validation training.

    Args:
        num_epochs (int): Number of epochs to train per fold.
    """
    set_seed(Config.SEED)
    logger = setup_logger(
        "MLCWNet_Training", os.path.join(Config.WORKING_DIR, "train.log")
    )

    # Retrieve K-Fold DataLoaders
    # load_cached_data=True allows using pre-processed data if available
    fold_loaders = get_kfold_loaders(load_cached_data=True)

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        logger.info(f"\n{'='*20} Fold {fold_idx + 1}/{Config.NUM_FOLDS} {'='*20}")
        train_fold(fold_idx, train_loader, val_loader, logger, num_epochs=num_epochs)


def generate_submission():
    """
    Generates submission file by ensembling predictions from all trained fold models.
    """
    set_seed(Config.SEED)
    logger = setup_logger(
        "MLCWNet_Inference", os.path.join(Config.WORKING_DIR, "inference.log")
    )
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    all_preds = []

    logger.info("Starting inference...")

    # Iterate through all folds
    for fold_idx in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold_idx}.pth")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping fold.")
            continue

        # Load Model
        model = MLCWNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []

        # Inference Loop
        with torch.no_grad():
            for images, angles, _ in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits)

                fold_preds.extend(probs.cpu().numpy().flatten())

        all_preds.append(fold_preds)
        logger.info(f"Fold {fold_idx + 1} inference complete.")

    if not all_preds:
        logger.error("No predictions generated. Check if models were trained.")
        return

    # Ensemble: Average Probabilities across folds
    all_preds = np.array(all_preds)
    avg_preds = np.mean(all_preds, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save to disk
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
