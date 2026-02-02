import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import CFG
from library.utils import (
    AverageMeter,
    get_score,
    calculate_class_weights,
    seed_everything,
)
from library.dataset import prepare_loaders
from library.model import AppleClassifier


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    losses = AverageMeter()
    start_time = time.time()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Convert one-hot/multi-label encoding to class indices for CrossEntropyLoss
        # Assuming mutually exclusive classes as per stratify_label logic
        targets = torch.argmax(labels, dim=1)

        batch_size = images.size(0)

        optimizer.zero_grad()

        y_preds = model(images)

        loss = criterion(y_preds, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch} - Train Loss: {losses.avg:.6f} - Time: {elapsed:.2f}s")

    return losses.avg


def valid_one_epoch(epoch, model, val_loader, criterion, device):
    """
    Validates the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on.

    Returns:
        tuple: (Average loss, ROC AUC score)
    """
    model.eval()

    losses = AverageMeter()
    preds_list = []
    labels_list = []
    start_time = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            targets = torch.argmax(labels, dim=1)
            batch_size = images.size(0)

            y_preds = model(images)
            loss = criterion(y_preds, targets)

            losses.update(loss.item(), batch_size)

            # Apply softmax to get probabilities for ROC AUC
            y_probs = torch.softmax(y_preds, dim=1)

            preds_list.append(y_probs.cpu().numpy())
            labels_list.append(
                labels.cpu().numpy()
            )  # Keep original one-hot for scoring

    elapsed = time.time() - start_time

    preds = np.concatenate(preds_list)
    targets = np.concatenate(labels_list)

    # get_score expects (y_true, y_pred)
    # y_true should be one-hot or binary indicators
    score = get_score(targets, preds)

    print(
        f"Epoch {epoch} - Val Loss: {losses.avg:.6f} - Val AUC: {score:.15f} - Time: {elapsed:.2f}s"
    )

    return losses.avg, score


def run_training(fold, backbone, debug=CFG.debug, epochs=CFG.epochs):
    """
    Orchestrates training for a specific fold and backbone.

    Args:
        fold (int): Fold index.
        backbone (str): Name of the backbone architecture.
        debug (bool): Whether to run in debug mode.
        epochs (int): Number of epochs to train.

    Returns:
        float: Best ROC AUC score achieved.
    """
    seed_everything(CFG.seed)

    device = CFG.device
    print(f"Starting training for Fold {fold}, Backbone: {backbone}")

    # Prepare DataLoaders
    train_loader, val_loader = prepare_loaders(fold, backbone, debug=debug)

    # Initialize Model
    model = AppleClassifier(backbone, pretrained=True)
    model.to(device)

    # Calculate Class Weights
    # We load the training metadata to compute weights based on the distribution
    train_meta = pd.read_csv(CFG.train_csv)
    weights = calculate_class_weights(train_meta).to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=CFG.min_lr)

    # Training Loop
    best_score = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)

        # Validate
        _, val_score = valid_one_epoch(epoch, model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            print(f"New Best Score: {best_score:.15f}. Saving model...")

            save_name = f"{backbone}_fold{fold}_best.pth"
            save_path = os.path.join(CFG.output_dir, save_name)
            torch.save(model.state_dict(), save_path)

            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= CFG.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Fold {fold} finished. Best ROC AUC: {best_score:.15f}")

    # Clear memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_score
