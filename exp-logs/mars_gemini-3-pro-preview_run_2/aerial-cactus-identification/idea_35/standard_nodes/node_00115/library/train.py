import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import CactusDataset
from library.model import UltraWideSERepNeXt


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Ensure labels shape matches outputs for BCEWithLogitsLoss
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for ROC AUC
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in the batch (unlikely in full val set but possible in debug)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def train_model(seed, epochs=None, debug=False):
    """
    Main function to train the model for a specific seed.

    Args:
        seed (int): Random seed for initialization.
        epochs (int, optional): Number of epochs to train. Defaults to Config.EPOCHS.
        debug (bool, optional): If True, uses a small subset of data for quick testing.

    Returns:
        float: Best validation ROC AUC achieved.
    """
    # 1. Reproducibility
    set_seed(seed)

    if epochs is None:
        epochs = Config.EPOCHS

    print(f"Starting training for Seed {seed}...")

    # 2. Data Loading
    train_dataset = CactusDataset(mode="train", load_cached_data=True)
    val_dataset = CactusDataset(mode="val", load_cached_data=True)

    if debug:
        # Use a small subset for debugging
        indices = list(range(min(len(train_dataset), 100)))
        train_dataset = Subset(train_dataset, indices)
        val_indices = list(range(min(len(val_dataset), 50)))
        val_dataset = Subset(val_dataset, val_indices)
        print("Debug mode: Using subset of data.")

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

    # 3. Model Initialization
    device = Config.DEVICE
    model = UltraWideSERepNeXt()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished for Seed {seed}. Best Val AUC: {best_val_auc}")
    return best_val_auc
