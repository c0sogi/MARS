import torch
import torch.nn as nn
import numpy as np
from library.utils import get_score
from library.config import Config


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        device (torch.device): The device to run on.
        epoch (int): The current epoch number.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Step scheduler if it's updated per iteration (e.g., OneCycleLR)
        # In this specific config, we use CosineAnnealingLR which is usually stepped per epoch,
        # but if a per-step scheduler were passed, it would go here.
        # For the proposed solution, the main loop handles the scheduler step if it's per epoch.

    epoch_loss = running_loss / dataset_size

    # Print full precision metric as requested
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): The device to run on.

    Returns:
        tuple: (validation_log_loss, predictions, true_labels)
            - validation_log_loss (float): The calculated Log Loss.
            - predictions (np.ndarray): The predicted probabilities (N, num_classes).
            - true_labels (np.ndarray): The ground truth labels (N,).
    """
    model.eval()

    all_preds = []
    all_labels = []

    # We use CrossEntropyLoss just to check consistency, but the main metric is calculated via get_score
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for Log Loss calculation
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    # Calculate metric using the utility function (sklearn log_loss)
    # This ensures we are optimizing for the exact competition metric
    val_log_loss = get_score(true_labels, predictions)

    # Also calculate the average CE loss (should be very close to log loss)
    avg_ce_loss = running_loss / dataset_size

    print(f"Validation Log Loss: {val_log_loss}")

    return val_log_loss, predictions, true_labels
