import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from library.utils import set_seed, get_device
from library.model import WideSEResNeXt
from library.dataset import CactusDataset, get_transforms


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    # Handle edge case where batch might contain only one class
    if len(np.unique(all_labels)) > 1:
        auc_score = roc_auc_score(all_labels, all_preds)
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training_cycle(
    seed,
    train_data,
    val_data,
    working_dir,
    batch_size=128,
    epochs=20,
    lr=1e-3,
    patience=5,
):
    """
    Runs the full training cycle for a specific seed.

    Args:
        seed (int): Random seed.
        train_data (tuple): (train_imgs, train_labels).
        val_data (tuple): (val_imgs, val_labels).
        working_dir (str): Directory to save checkpoints.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.

    Returns:
        str: Path to the best saved model checkpoint.
    """
    # 1. Reproducibility
    set_seed(seed)
    device = get_device()

    # 2. Data Preparation
    train_imgs, train_labels = train_data
    val_imgs, val_labels = val_data

    train_dataset = CactusDataset(
        train_imgs, train_labels, transform=get_transforms(split="train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_labels, transform=get_transforms(split="val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = WideSEResNeXt(num_classes=1).to(device)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(working_dir, f"model_seed_{seed}.pth")

    print(f"Starting training for Seed {seed}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_model_path
