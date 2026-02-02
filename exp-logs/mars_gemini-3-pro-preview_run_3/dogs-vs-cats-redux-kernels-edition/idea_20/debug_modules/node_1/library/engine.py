import torch
import torch.nn as nn
import numpy as np
import copy
from library.config import Config
from library.utils import get_device


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer for weight updates.
        device (torch.device): The device (CPU/GPU) to run on.
        epoch (int): The current epoch number (0-indexed).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Binary Cross Entropy with Logits for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in dataloader:
        images = images.to(device)
        # Ensure labels are float and have shape [Batch, 1]
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to run on.

    Returns:
        tuple: (average_loss, predictions, true_labels)
               predictions are probabilities after sigmoid.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])

    return avg_loss, all_preds, all_labels


def predict(model, dataloader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Strategy: Average of predictions from original image and horizontally flipped image.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set (returns image, id).
        device (torch.device): The device to run on.

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Forward pass with original images
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass with horizontally flipped images (TTA)
            # Flip along width dimension (dim 3 for NCHW format)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # 3. Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_probs.append(avg_probs.cpu().numpy().flatten())
            all_ids.append(ids.numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_ids = np.concatenate(all_ids)
    else:
        all_probs = np.array([])
        all_ids = np.array([])

    return all_ids, all_probs


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
):
    """
    Orchestrates the training process with Early Stopping and Scheduler management.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler (stepped per epoch).
        device (torch.device): Device.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience count.

    Returns:
        tuple: (best_model, best_val_loss)
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_loss, _, _ = evaluate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)

    return model, best_loss
