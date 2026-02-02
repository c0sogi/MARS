import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import AverageMeter, calculate_log_loss, save_checkpoint
from library.model import SelectiveSECNN


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Handles the training of one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, angles, labels) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Forward pass
        # Model expects (x, angle)
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Handles validation logic.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run validation on.

    Returns:
        tuple: (average_loss, log_loss_metric)
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to logits to get probabilities for metric calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Log Loss using the utility function
    metric = calculate_log_loss(all_targets, all_preds)

    return losses.avg, metric


def train_fold(fold_idx, train_loader, val_loader):
    """
    Manages the training loop for a specific cross-validation fold.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.

    Returns:
        model (nn.Module): The model with the best weights loaded.
        best_score (float): The best validation log loss achieved.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = SelectiveSECNN()
    model = model.to(device)

    # Initialize Optimizer
    # Adam with constant learning rate as per Idea
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Loss Function
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_score = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} | Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val LogLoss: {val_metric} | "
            f"Time: {elapsed}s"
        )

        # Checkpoint and Early Stopping Logic
        is_best = val_metric < best_score

        if is_best:
            best_score = val_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                fold=fold_idx,
            )
        else:
            patience_counter += 1
            # Save checkpoint (not best)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                fold=fold_idx,
            )

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}. Best score: {best_score} at epoch {best_epoch}"
            )
            break

    # Load best weights before returning
    best_checkpoint_path = f"{Config.CHECKPOINT_DIR}/model_best_fold_{fold_idx}.pth"
    print(f"Loading best weights from {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model, best_score
