import os
import torch
import numpy as np
from library.utils import calculate_accuracy


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler (stepped per batch).
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for bags, masks, targets in loader:
        bags = bags.to(device)
        masks = masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(bags, masks)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate loss (multiply by batch size to get total loss for batch)
        batch_size = bags.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bags, masks, targets in loader:
            bags = bags.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            outputs = model(bags, masks)
            loss = criterion(outputs, targets)

            batch_size = bags.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Collect predictions for accuracy calculation
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = calculate_accuracy(all_preds, all_targets)

    return avg_loss, accuracy


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    patience,
    save_path,
    device,
):
    """
    Runs the full training loop with early stopping and checkpointing.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience (epochs without improvement).
        save_path (str): Path to save the best model weights.
        device (str): Device to run on.

    Returns:
        float: Best validation accuracy achieved.
    """
    best_acc = 0.0
    patience_counter = 0

    # Ensure save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        # Checkpointing and Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Validation Accuracy: {best_acc}")
    return best_acc
