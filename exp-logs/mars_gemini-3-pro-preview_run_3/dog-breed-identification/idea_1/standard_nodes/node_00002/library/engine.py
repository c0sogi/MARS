import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import get_device, seed_everything


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to compute on.
        scaler (GradScaler): AMP GradScaler.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to compute on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    device,
    scheduler=None,
    save_path="./working/best_model.pth",
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        device (torch.device): Device.
        scheduler (lr_scheduler, optional): Learning rate scheduler.
        save_path (str): Path to save the best model weights.

    Returns:
        nn.Module: The model with the best validation weights loaded.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_loss = float("inf")
    epochs_no_improve = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(
        f"Starting training on {device} for {num_epochs} epochs with patience {patience}..."
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_loss}"
                )
                break

    # Load best model weights
    if os.path.exists(save_path):
        print(f"Loading best model weights from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test dataloader.
        device (torch.device): Device.

    Returns:
        tuple: (ids_list, probabilities_numpy_array)
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Apply Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, np.concatenate(all_probs, axis=0)


def save_submission(ids, probs, class_names, output_path="./submission/submission.csv"):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        ids (list): List of image IDs.
        probs (np.ndarray): Array of predicted probabilities (N_samples, N_classes).
        class_names (list): List of class names corresponding to probability columns.
        output_path (str): Path to save the submission CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
