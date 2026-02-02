import torch
import torch.nn as nn
from library.utils import calculate_roc_auc
from library.dataset import mixup_data


def train_one_epoch(model, dataloader, optimizer, device, pos_weight):
    """
    Trains the model for one epoch using Mixup augmentation and class-weighted BCE loss.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer for updating weights.
        device (torch.device): The device (CPU or GPU) to use.
        pos_weight (torch.Tensor): Weights for positive examples for BCEWithLogitsLoss.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    # Initialize loss function with positive weights for class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        # alpha=0.4 is the setting mentioned in the idea description
        images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=0.4)

        # Forward pass
        outputs = model(images)

        # Calculate Mixup Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device, pos_weight):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device (CPU or GPU) to use.
        pos_weight (torch.Tensor): Weights for positive examples for BCEWithLogitsLoss.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0

    all_targets = []
    all_preds = []

    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)

            # Calculate Loss (No Mixup for validation)
            loss = criterion(logits, targets)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    avg_loss = running_loss / len(dataloader)

    # Concatenate all batches
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Calculate Metric
    auc_score = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc_score
