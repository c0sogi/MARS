import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils, data, model


def train_one_epoch(train_loader, model_instance, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.

    Args:
        train_loader: DataLoader for training data.
        model_instance: The neural network model.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device (cpu or cuda).
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model_instance.train()
    losses = utils.AverageMeter()

    for i, (images, angles, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).view(-1, 1)

        # Forward pass
        outputs = model_instance(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model_instance, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader: DataLoader for validation data.
        model_instance: The neural network model.
        criterion: Loss function.
        device: Torch device.

    Returns:
        float: Average validation loss.
    """
    model_instance.eval()
    losses = utils.AverageMeter()

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model_instance(images, angles)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_fold(fold_idx):
    """
    Orchestrates the training process for a single fold.
    Initializes model, optimizer, and handles early stopping.

    Args:
        fold_idx (int): The index of the current fold (0 to NUM_FOLDS-1).

    Returns:
        float: The best validation loss achieved for this fold.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED + fold_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Fold {fold_idx} on {device}...")

    # Load Data
    train_loader, val_loader = data.get_dataloaders(fold_idx, load_cached_data=True)

    # Initialize Model (SPPCNN)
    net = model.SPPCNN().to(device)

    # Initialize Optimizer (AdamW)
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Loss Function (BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            train_loader, net, criterion, optimizer, device, epoch
        )

        # Validation Step
        val_loss = validate(val_loader, net, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision for val_loss as requested)
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping and Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            # print(f"Fold {fold_idx} | New Best Val Loss: {best_val_loss}")
        else:
            patience_counter += 1

        # Save Checkpoint
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "fold_idx": fold_idx,
            },
            is_best,
            fold_idx,
        )

        if patience_counter >= config.PATIENCE:
            print(f"Fold {fold_idx} | Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Fold {fold_idx} Finished. Best Val Loss: {best_val_loss}")
    return best_val_loss


def run_training():
    """
    Runs the training loop for all configured folds.
    """
    fold_losses = []
    for fold_idx in range(config.NUM_FOLDS):
        best_loss = train_fold(fold_idx)
        fold_losses.append(best_loss)

    print("\nAll Folds Finished.")
    print(f"CV Scores: {fold_losses}")
    print(f"Average CV Loss: {np.mean(fold_losses)}")
