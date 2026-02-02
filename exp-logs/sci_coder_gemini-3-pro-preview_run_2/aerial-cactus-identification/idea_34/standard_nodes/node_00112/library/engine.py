import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.utils import calculate_roc_auc, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (str): Device to run on.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        # Unpack batch. Dataset returns (image, label, id) for train
        images, labels, _ = batch

        images = images.to(device)
        # BCEWithLogitsLoss expects target shape (N, 1) to match output (N, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate(model, dataloader, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation dataloader.
        criterion (Loss): Loss function.
        device (str): Device to run on.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            # Unpack batch. Dataset returns (image, label, id) for val
            images, labels, _ = batch

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    # Concatenate results for AUC calculation
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        auc = calculate_roc_auc(all_targets, all_preds)
    else:
        auc = 0.5

    return avg_loss, auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    filename="model.pth",
    max_batches=None,
):
    """
    Main training loop with early stopping.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device.
        num_epochs (int): Max epochs.
        patience (int): Early stopping patience.
        filename (str): Filename to save best model.
        max_batches (int, optional): Debugging limit.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    best_val_auc = -1.0
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        # Train phase
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches
        )

        # Validation phase
        val_loss, val_auc = evaluate(model, val_loader, criterion, device, max_batches)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        # We maximize AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save checkpoint
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_val_auc,
            }
            save_checkpoint(state, filename)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc}")


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test dataloader.
        device (str): Device.

    Returns:
        tuple: (ids, probabilities)
            ids: List of image IDs.
            probabilities: List of predicted probabilities (floats).
    """
    model.eval()
    model.to(device)

    ids_list = []
    probs_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch. Dataset returns (image, id) for test (no labels)
            images, ids = batch

            images = images.to(device)

            # Forward
            outputs = model(images)

            # Sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            ids_list.extend(ids)
            probs_list.extend(probs.cpu().numpy().flatten().tolist())

    return ids_list, probs_list
