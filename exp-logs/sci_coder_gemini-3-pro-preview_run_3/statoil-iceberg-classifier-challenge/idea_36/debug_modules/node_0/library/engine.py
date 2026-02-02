import torch
import numpy as np
from library.config import Config


def update_dropblock_prob(model, epoch, total_epochs):
    """
    Updates the DropBlock probability for the model based on the current epoch.
    Implements a linear schedule from 0 to Config.DROPBLOCK_PROB.

    Args:
        model (torch.nn.Module): The model containing DropBlock layers.
        epoch (int): The current epoch number (0-indexed).
        total_epochs (int): Total number of epochs to train.
    """
    if total_epochs <= 0:
        prob = 0.0
    else:
        # Linear schedule: increases linearly as training progresses
        prob = Config.DROPBLOCK_PROB * (epoch / total_epochs)

    # Clamp to ensure it doesn't exceed target probability
    prob = min(prob, Config.DROPBLOCK_PROB)

    # Update the model's DropBlock layers if the method exists
    if hasattr(model, "set_drop_prob"):
        model.set_drop_prob(prob)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (callable): The loss function.
        device (torch.device): The device to train on.
        epoch (int): Current epoch number.
        total_epochs (int): Total epochs.

    Returns:
        float: The average loss for this epoch.
    """
    model.train()

    # Update DropBlock regularization strength
    update_dropblock_prob(model, epoch, total_epochs)

    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Move data to target device
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    # Calculate average loss
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): The validation data loader.
        criterion (callable): The loss function.
        device (torch.device): The device to evaluate on.

    Returns:
        float: The average validation loss (Log Loss).
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            batch_size = images.size(0)

            # Forward pass
            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return val_loss


def predict(model, loader, device):
    """
    Generates predictions for the dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (ids, probabilities) - numpy arrays of IDs and predicted probabilities.
    """
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            outputs = model(images, angles)

            # Apply sigmoid to convert logits to probabilities (0-1)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy())
            all_ids.extend(batch_ids)

    return np.array(all_ids), np.array(all_preds)
