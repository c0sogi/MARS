import torch
import torch.nn as nn
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (nn.Module): Loss function.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        # Model output is logits (before sigmoid)
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimization step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, all_probabilities, all_labels)
               - average_loss (float): Mean loss over validation set.
               - all_probabilities (np.ndarray): Flattened array of predicted probabilities.
               - all_labels (np.ndarray): Flattened array of ground truth labels.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten for metric calculation
            # We flatten here to reduce memory overhead of keeping tensor shapes
            # if we only care about pixel-wise metrics.
            all_probs.append(probs.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if all_probs:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
    else:
        all_probs = np.array([])
        all_labels = np.array([])

    return avg_loss, all_probs, all_labels


def calculate_fbeta(preds, labels, beta=0.5):
    """
    Calculates the F-beta score.

    Args:
        preds (np.ndarray): Binary predictions (0 or 1).
        labels (np.ndarray): Binary ground truth labels (0 or 1).
        beta (float): Beta value for F-score (default 0.5).

    Returns:
        float: F-beta score.
    """
    # True Positives, False Positives, False Negatives
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F-beta score
    if precision + recall == 0:
        return 0.0

    beta_sq = beta**2
    f_beta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

    return f_beta


def optimize_threshold(probs, labels):
    """
    Finds the best threshold to maximize the F0.5 score.

    Args:
        probs (np.ndarray): Predicted probabilities (0.0 to 1.0).
        labels (np.ndarray): Ground truth labels (0 or 1).

    Returns:
        tuple: (best_threshold, best_f05_score)
    """
    best_threshold = 0.5
    best_score = 0.0

    # Define search range from Config
    start = Config.THRESHOLD_SEARCH_START
    end = Config.THRESHOLD_SEARCH_END
    step = Config.THRESHOLD_SEARCH_STEP

    # Generate thresholds
    thresholds = np.arange(start, end + step, step)

    # Iterate through thresholds
    for t in thresholds:
        preds = (probs >= t).astype(np.uint8)
        score = calculate_fbeta(preds, labels, beta=0.5)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score
