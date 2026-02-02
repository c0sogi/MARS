import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, mixup_data, mixup_criterion, save_checkpoint


def train_one_epoch(model, optimizer, dataloader, device, epoch, max_steps=None):
    """
    Trains the model for one epoch using Mixup augmentation.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        dataloader: Training dataloader.
        device: Computation device.
        epoch: Current epoch number (for logging).
        max_steps: Optional limit on steps per epoch (for debugging).

    Returns:
        float: Average training loss.
    """
    model.train()
    loss_meter = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for step, (images, targets) in enumerate(dataloader):
        if max_steps and step >= max_steps:
            break

        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        outputs = model(images)
        # Ensure outputs are (B,) to match targets
        outputs = outputs.view(-1)

        # Compute Loss using Mixup criterion
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg:.8f}")
    return loss_meter.avg


def validate_one_epoch(model, dataloader, device, max_steps=None):
    """
    Evaluates the model on the validation set using raw Log Loss.

    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        device: Computation device.
        max_steps: Optional limit on steps (for debugging).

    Returns:
        float: Average validation loss.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            if max_steps and step >= max_steps:
                break

            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1)

            # Calculate raw loss (no mixup)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {loss_meter.avg}")
    return loss_meter.avg


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    epochs,
    patience,
    checkpoint_name,
):
    """
    Executes the full training loop with Early Stopping.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        device: Computation device.
        epochs: Total number of epochs.
        patience: Epochs to wait for improvement before stopping.
        checkpoint_name: Filename for saving the checkpoint.

    Returns:
        float: Best validation loss achieved.
    """
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss = validate_one_epoch(model, val_loader, device)

        # Step Scheduler (assuming epoch-based scheduler like CosineAnnealingLR)
        if scheduler is not None:
            scheduler.step()

        # Checkpoint & Early Stopping Logic
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_loss": best_loss,
                "val_loss": val_loss,
            },
            is_best,
            checkpoint_name,
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_loss


def predict(model, dataloader, device):
    """
    Generates predictions for the test set. Handles Test-Time Augmentation (TTA)
    if enabled in Config.

    Args:
        model: PyTorch model.
        dataloader: Test dataloader (returns images and ids).
        device: Computation device.

    Returns:
        ids (list): List of image IDs.
        probs (list): List of predicted probabilities.
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # Original prediction
            logits = model(images)
            probs = torch.sigmoid(logits).view(-1)

            if Config.TTA_FLIP:
                # Horizontal Flip TTA
                # Flip on width dimension (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped).view(-1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_ids.extend(ids.tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_ids, all_probs


def generate_submission(model, dataloader, device, save_path=Config.SUBMISSION_PATH):
    """
    Generates predictions and saves the submission CSV file.

    Args:
        model: PyTorch model.
        dataloader: Test dataloader.
        device: Computation device.
        save_path: Path to save the CSV.
    """
    ids, probs = predict(model, dataloader, device)

    df = pd.DataFrame({"id": ids, "label": probs})

    # Ensure consistent sorting by ID
    df = df.sort_values("id")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
