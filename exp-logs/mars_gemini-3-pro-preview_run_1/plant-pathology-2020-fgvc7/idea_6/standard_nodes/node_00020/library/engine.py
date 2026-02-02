import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_logger
from library.dataset import mixup, cutmix

logger = get_logger("engine")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, data in enumerate(dataloader):
        images = data["image"].to(device)
        targets = data["target"].to(device)

        batch_size = images.size(0)

        # Apply Mixup or CutMix with probability Config.MIXUP_PROB
        # We assume targets are already one-hot/float from the dataset (B, N_Classes)
        if np.random.rand() < Config.MIXUP_PROB:
            # Randomly choose between Mixup and CutMix
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup(
                    images, targets, Config.MIXUP_ALPHA
                )
            else:
                images, targets_a, targets_b, lam = cutmix(
                    images, targets, Config.CUTMIX_ALPHA
                )

            # Create soft targets
            # targets shape is (B, 4), so we can interpolate directly
            targets = lam * targets_a + (1 - lam) * targets_b

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        # WeightedSoftCrossEntropy handles soft targets (B, C) directly
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run on.

    Returns:
        tuple: (average_loss, average_roc_auc)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            images = data["image"].to(device)
            targets = data["target"].to(device)

            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for AUC
            preds = torch.softmax(outputs, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate ROC AUC
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    # This matches "Mean column-wise ROC AUC".
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError as e:
        # Handle edge cases where a class might not be present in the validation batch
        logger.warning(f"ROC AUC calculation failed: {e}. Returning 0.0.")
        epoch_auc = 0.0

    return epoch_loss, epoch_auc
