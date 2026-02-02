import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_roc_auc, save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        # Ensure labels are (B, 1) float tensors for BCEWithLogitsLoss
        labels = batch["label"].to(device).unsqueeze(1)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        auc_score = calculate_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Original + HFlip + VFlip).

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.
        device (str): 'cuda' or 'cpu'.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'has_cactus' probabilities.
    """
    model.eval()
    results = {"id": [], "has_cactus": []}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            ids = batch["id"]

            # 1. Original Image
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average predictions
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            results["id"].extend(ids)
            results["has_cactus"].extend(avg_prob.cpu().numpy().flatten())

    return pd.DataFrame(results)


def train_engine(model, train_loader, val_loader, optimizer, scheduler, device, seed):
    """
    Orchestrates the training process for a single model instance (seed).
    Handles training loops, validation, logging, and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device to train on.
        seed (int): Current seed (for saving checkpoints).

    Returns:
        float: The best validation AUC achieved.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    print(f"Starting training for Seed {seed}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Val AUC for Seed {seed}: {best_auc}")
    return best_auc
