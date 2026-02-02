import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, setup_logger, save_checkpoint, AverageMeter
from library.data_loader import get_loaders
from library.model import RTICNN


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The RTI-CNN model.
        criterion (nn.Module): Loss function (BCEWithLogitsLoss).
        optimizer (Optimizer): Optimizer (AdamW).
        device (str): Device to run on ('cuda' or 'cpu').
        epoch (int): Current epoch number.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accuracies = AverageMeter("Acc", ":6.2f")

    model.train()
    end = time.time()

    for i, (images, angles, targets) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        # Forward pass
        # Model expects (x, angle)
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Compute accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == targets).float().mean() * 100.0

        # Record metrics
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, accuracies.avg


def validate(val_loader, model, criterion, device, logger):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The RTI-CNN model.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    losses = AverageMeter("Loss", ":.4e")
    accuracies = AverageMeter("Acc", ":6.2f")

    model.eval()

    with torch.no_grad():
        for i, (images, angles, targets) in enumerate(val_loader):
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == targets).float().mean() * 100.0

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))

    return losses.avg, accuracies.avg


def run_fold(fold_idx):
    """
    Runs the training pipeline for a single fold.

    Args:
        fold_idx (int): Index of the current fold (0-4).

    Returns:
        float: Best validation loss achieved.
    """
    # Setup
    set_seed(Config.SEED + fold_idx)  # Ensure distinct but reproducible seeds per fold
    device = torch.device(Config.DEVICE)

    # Logger
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold_{fold_idx}.log")
    logger = setup_logger(f"Fold_{fold_idx}", log_file)
    logger.info(f"Starting Fold {fold_idx}")

    # Data Loaders
    train_loader, val_loader = get_loaders(fold_idx, load_cached_data=True)

    # Model
    model = RTICNN().to(device)

    # Optimizer & Loss
    # Constant LR as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_acc = validate(val_loader, model, criterion, device, logger)

        # Logging (Full Precision)
        logger.info(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}%, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}%"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(
                f"New best validation loss: {best_val_loss}. Saving checkpoint."
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                fold_idx=fold_idx,
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss}"
                )
                break

    return best_val_loss
