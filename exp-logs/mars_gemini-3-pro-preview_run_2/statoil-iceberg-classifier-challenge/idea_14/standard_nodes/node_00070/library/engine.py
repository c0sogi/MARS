import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from library.model import DPCNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The DPCNet model.
        loader (DataLoader): Training data loader.
        criterion (loss): Loss function.
        optimizer (optim): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets in loader:
        img, angle = inputs
        img = img.to(device)
        angle = angle.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model((img, angle))
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The DPCNet model.
        loader (DataLoader): Validation data loader.
        criterion (loss): Loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets in loader:
            img, angle = inputs
            img = img.to(device)
            angle = angle.to(device)
            targets = targets.to(device)

            outputs = model((img, angle))
            loss = criterion(outputs, targets)

            running_loss += loss.item() * img.size(0)

    return running_loss / dataset_size


def train_fold(
    fold_idx,
    train_loader,
    val_loader,
    device,
    epochs=60,
    learning_rate=2e-4,
    patience=10,
    output_dir="./working",
):
    """
    Trains a single fold of the DPCNet model.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run on.
        epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        patience (int): Early stopping patience.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        tuple: (path_to_best_model, best_val_loss)
    """
    print(f"\nStarting Fold {fold_idx}")

    model = DPCNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Cite debug_lesson_2: Remove deprecated 'verbose' argument from PyTorch schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Print full precision metrics as requested
        print(f"  Epoch {epoch+1}: Train Loss {train_loss}, Val Loss {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"  Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss}"
            )
            break

    # Save the best model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"dpcnet_fold_{fold_idx}.pth")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, save_path)
    else:
        # Fallback if training failed to improve (saves last state)
        torch.save(model.state_dict(), save_path)

    print(f"Fold {fold_idx} model saved to {save_path}")
    return save_path, best_val_loss
