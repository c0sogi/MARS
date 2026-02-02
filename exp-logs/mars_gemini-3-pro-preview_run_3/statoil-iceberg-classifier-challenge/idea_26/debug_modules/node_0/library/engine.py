import torch
import numpy as np
from library.config import Config


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        # Squeeze logits to match label shape (B,)
        logits = logits.squeeze(1)

        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # Calculate accuracy
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images, angles)
            logits = logits.squeeze(1)

            loss = criterion(logits, labels)

            # Accumulate metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # Calculate accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to run on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).squeeze(1)
            preds_list.extend(probs.cpu().numpy())

    return np.array(preds_list)
