import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import library.config as config
import library.utils as utils
import library.data as data
import library.model as model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The computing device (CPU/GPU).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The computing device.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Calculate ROC AUC
    # Handle edge case where batch might contain only one class
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
):
    """
    Orchestrates the training process including data loading, model initialization,
    training loop, validation, and early stopping.
    """
    # 1. Reproducibility
    utils.seed_everything(config.SEED)

    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # 3. Datasets and DataLoaders
    train_dataset = data.BraTSDataset(
        metadata=df_train,
        base_dir=config.INPUT_DIR,
        transform=data.get_transforms("train"),
        is_test=False,
    )

    val_dataset = data.BraTSDataset(
        metadata=df_val,
        base_dir=config.INPUT_DIR,
        transform=data.get_transforms("val"),
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model, Optimizer, Loss
    print("Initializing model...")
    net = model.MGMTNet(pretrained=True)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_loss, val_auc = evaluate(net, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(net.state_dict(), config.CHECKPOINT_PATH)
            print(f"Validation loss improved. Model saved to {config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
