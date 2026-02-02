import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import calculate_accuracy, get_logger, seed_everything
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaClassifier


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0

    start_time = time.time()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Calculate metrics
        batch_size = images.size(0)
        acc = calculate_accuracy(outputs, labels)

        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples
    epoch_acc = running_acc / num_samples
    duration = time.time() - start_time

    logger.info(
        f"Epoch {epoch} Training - Loss: {epoch_loss}, Accuracy: {epoch_acc}, Time: {duration}s"
    )

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, logger):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size

    epoch_loss = running_loss / num_samples
    epoch_acc = running_acc / num_samples
    duration = time.time() - start_time

    # Print full precision as requested
    logger.info(
        f"Validation - Loss: {epoch_loss}, Accuracy: {epoch_acc}, Time: {duration}s"
    )

    return epoch_loss, epoch_acc


def run_training(subset_size=None, epochs=Config.EPOCHS, patience=3):
    """
    Main training loop orchestration.

    Args:
        subset_size (int, optional): Number of samples to use (for debugging).
        epochs (int): Number of epochs to train.
        patience (int): Early stopping patience.
    """
    # Setup Logger
    logger = get_logger(Config.LOG_PATH)
    logger.info("Starting training process...")

    # Set Seed for reproducibility
    seed_everything(Config.SEED)

    # Device
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # Data Loaders
    logger.info("Initializing Datasets and DataLoaders...")
    train_dataset = CassavaDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        data_split="train",
        subset_size=subset_size,
    )

    val_dataset = CassavaDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_transforms("val"),
        data_split="val",
        subset_size=subset_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = CassavaClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.MIN_LR
    )

    # Loss Function (CrossEntropy with Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # Training Loop
    best_acc = 0.0
    patience_counter = 0

    logger.info("Beginning training loop...")

    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Current Learning Rate: {current_lr}")

        # Checkpointing and Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info(
                f"Validation accuracy improved ({best_acc}). Saving model to {Config.MODEL_CHECKPOINT_PATH}..."
            )
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"No improvement in validation accuracy. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc}")
    return best_acc
