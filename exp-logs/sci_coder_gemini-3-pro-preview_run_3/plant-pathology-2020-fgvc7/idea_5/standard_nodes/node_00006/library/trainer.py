import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.dataset import get_dataloaders
from library.models import MultiLevelEfficientNet, SwinTransformerModel
from library.utils import get_class_weights, calculate_metric, seed_everything


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on (CPU/GPU).

    Returns:
        tuple: (epoch_loss, epoch_auc)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # The dataset returns one-hot encoded labels (float32).
        # nn.CrossEntropyLoss expects class indices (LongTensor) as targets.
        target_indices = torch.argmax(labels, dim=1)

        loss = criterion(outputs, target_indices)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Apply softmax to get probabilities for AUC calculation
        probs = torch.softmax(outputs, dim=1)

        all_preds.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches for metric calculation
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_auc = calculate_metric(all_labels, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (val_loss, val_auc)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            target_indices = torch.argmax(labels, dim=1)
            loss = criterion(outputs, target_indices)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    val_auc = calculate_metric(all_labels, all_preds)

    return val_loss, val_auc


def train_fold(
    fold_idx: int,
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = Config.EPOCHS,
    debug: bool = False,
):
    """
    Trains a single fold for a specific model architecture.

    Args:
        fold_idx (int): The index of the current fold.
        model_type (str): 'effnet' or 'swin'.
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        epochs (int): Number of epochs to train.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        float: The best validation AUC achieved.
    """
    # Set seed for this fold to ensure reproducibility
    seed_everything(Config.SEED + fold_idx)
    device = Config.DEVICE

    # Handle Debugging
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(
            f"Debug mode: Training on {len(train_df)} samples, Validating on {len(val_df)} samples."
        )

    # Initialize Model based on type
    if model_type == "effnet":
        model = MultiLevelEfficientNet(pretrained=True)
        img_size = Config.IMG_SIZE_EFFNET
    elif model_type == "swin":
        model = SwinTransformerModel(pretrained=True)
        img_size = Config.IMG_SIZE_SWIN
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, img_size, Config.BATCH_SIZE
    )

    # Define Loss Function with Class Weights
    if Config.USE_CLASS_WEIGHTS:
        weights = get_class_weights(train_df)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"{model_type}_fold_{fold_idx}_best.pth"
    )

    print(f"Starting training for Fold {fold_idx} ({model_type})...")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Print metrics (Full precision for Val AUC as requested)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Fold {fold_idx} finished. Best Val AUC: {best_auc}")
    return best_auc
