import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.model import train_one_epoch, validate


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    fold_idx,
    epochs=50,
    patience=10,
    lr=1e-3,
    weight_decay=1e-4,
    checkpoint_dir="./working/idea_15/checkpoints",
):
    """
    Manages the training lifecycle for a single fold.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device (CPU/GPU) to use for training.
        fold_idx (int): The index of the current fold (for logging and saving).
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate for the Adam optimizer.
        weight_decay (float): L2 regularization factor.
        checkpoint_dir (str): Directory to save the best model checkpoint.

    Returns:
        float: The best validation loss achieved.
        str: The file path to the saved best model.
    """

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Criterion (Loss Function)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize Optimizer (Adam with constant LR and Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")

    for epoch in range(epochs):
        # Perform one training epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Perform validation
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return best_loss, best_model_path
