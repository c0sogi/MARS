import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for mixed inputs
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (str): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader.
        criterion (Loss): Loss function.
        device (str): Device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Macro AUC
    # Handle potential errors if a class is missing in the validation set
    # We calculate per-class AUC and average only valid ones to handle degenerate batches (Cite debug_lesson_4)
    class_aucs = []
    for i in range(all_targets.shape[1]):
        # Only calculate AUC if the class has both positive and negative samples
        if len(np.unique(all_targets[:, i])) == 2:
            try:
                class_aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
            except ValueError:
                pass

    if class_aucs:
        auc = np.mean(class_aucs)
    else:
        # Fallback if no classes are valid (e.g., extremely small batch with no positives)
        auc = 0.5

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation AUC: {auc}")

    return avg_loss, auc


def predict(model, loader, device):
    """
    Generates predictions for the given loader.

    Args:
        model (nn.Module): The model to use.
        loader (DataLoader): Dataloader to predict on.
        device (str): Device to run on.

    Returns:
        np.ndarray: Array of probabilities (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
