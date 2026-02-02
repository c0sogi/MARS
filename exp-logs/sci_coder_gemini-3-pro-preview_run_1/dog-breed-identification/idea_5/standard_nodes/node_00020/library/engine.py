import torch
import torch.nn as nn
import time
from library.config import Config
from library.utils import get_logger, AverageMeter

# Initialize logger
logger = get_logger(name="engine")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def run_two_phase_training(model, train_loader, val_loader, save_path):
    """
    Orchestrates the two-phase training strategy:
    1. Head Adaptation: Freeze backbone, train head.
    2. Fine-Tuning: Unfreeze backbone, use discriminative learning rates.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        save_path (str): Path to save the best model checkpoint.
    """
    device = Config.DEVICE
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")

    # -------------------------------------------------------------------------
    # PHASE 1: Head Adaptation
    # -------------------------------------------------------------------------
    logger.info("Starting Phase 1: Head Adaptation (Freezing Backbone)")

    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Ensure head parameters are trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Optimizer for head only
    optimizer_phase1 = torch.optim.AdamW(
        model.head.parameters(),
        lr=Config.LR_HEAD_INIT,
        weight_decay=Config.WEIGHT_DECAY,
    )

    for epoch in range(1, Config.EPOCHS_HEAD + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_phase1, device, epoch
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"[Phase 1 Epoch {epoch}/{Config.EPOCHS_HEAD}] "
            f"Train Loss: {train_loss} Val Loss: {val_loss}"
        )

        # Save if best (even during phase 1, though unlikely to be final best)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with loss: {best_loss}")

    # -------------------------------------------------------------------------
    # PHASE 2: Fine-Tuning
    # -------------------------------------------------------------------------
    logger.info("Starting Phase 2: Fine-Tuning (Unfreezing Backbone)")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Discriminative Learning Rates
    # Lower LR for backbone to preserve features, higher LR for head
    optimizer_phase2 = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": model.head.parameters(), "lr": Config.LR_HEAD_FINE},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    for epoch in range(1, Config.EPOCHS_FINE + 1):
        # Current global epoch count
        global_epoch = Config.EPOCHS_HEAD + epoch

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_phase2, device, global_epoch
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"[Phase 2 Epoch {epoch}/{Config.EPOCHS_FINE}] "
            f"Train Loss: {train_loss} Val Loss: {val_loss}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with loss: {best_loss}")

    logger.info(f"Training completed. Best Validation Loss: {best_loss}")

    # Load best weights before returning
    logger.info(f"Loading best weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    return model
