import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
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
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        data_loader: Training DataLoader.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Standard BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        # Using alpha=1.0 as a standard default for Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, alpha=1.0, device=device
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def validate_one_epoch(model, data_loader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        data_loader: Validation DataLoader.
        device: Device to validate on.

    Returns:
        tuple: (average_loss, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate ROC AUC
    # Note: calculate_roc_auc handles cases where a class might be missing in the batch/set
    auc_score = calculate_roc_auc(targets, predictions)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation ROC AUC: {auc_score}")

    return avg_loss, predictions, targets
