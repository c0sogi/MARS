import torch
import numpy as np
from library.config import Config
from library.utils import calculate_robust_roc_auc


def mixup_data(x, y, alpha=0.4, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the data.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss using a linear combination of losses.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using standard optimizer and Mixup.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Calculation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, alpha=0.4, device=device
        )

        # Forward pass
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # Backward and optimize
        optimizer.zero_grad()
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
        model: PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Calculation device.

    Returns:
        tuple: (Average Validation Loss, Robust ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        auc_score = calculate_robust_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return avg_loss, auc_score
