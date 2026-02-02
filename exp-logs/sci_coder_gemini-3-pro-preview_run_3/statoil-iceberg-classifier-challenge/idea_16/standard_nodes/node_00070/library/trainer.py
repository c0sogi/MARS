import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import save_checkpoint


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for i, (images, angles, labels) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # [Batch] -> [Batch, 1]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, predictions, targets)
            - average_loss (float): The Log Loss on the validation set.
            - predictions (np.ndarray): Predicted probabilities.
            - targets (np.ndarray): True labels.
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(loader)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            logits = model(images, angles)

            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches

    # Concatenate all batches
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    return avg_loss, predictions, targets


def train_fold(fold_idx, model, train_loader, val_loader, device):
    """
    Runs the full training loop for a specific fold, including early stopping.

    Args:
        fold_idx (int): The current fold index.
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): Training loader.
        val_loader (torch.utils.data.DataLoader): Validation loader.
        device (str): Device to run on.

    Returns:
        float: The best validation loss achieved.
    """
    print(f"\n--- Starting Training for Fold {fold_idx} ---")

    # Define Optimizer and Loss
    # Using Adam with constant learning rate and weight decay as per Idea 16
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, _, _ = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  New best validation loss: {best_val_loss:.10f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            fold_idx,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss:.10f}")
    return best_val_loss
