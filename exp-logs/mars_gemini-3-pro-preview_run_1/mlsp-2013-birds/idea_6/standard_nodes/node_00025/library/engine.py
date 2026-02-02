import numpy as np
import torch
import torch.nn as nn
from library.utils import calculate_roc_auc


def train_one_epoch(model, loader, optimizer, device, config):
    """
    Trains the model for one epoch using Mixup augmentation.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Device to train on.
        config (Config): Configuration object containing hyperparameters.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Mixup Augmentation
        if config.MIXUP_ALPHA > 0:
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            # Forward pass
            logits = model(mixed_images)
            loss = criterion(logits, mixed_labels)
        else:
            # Standard training without Mixup
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Validation data loader.
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            # Validation loader returns (images, labels)
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        auc_score = calculate_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return avg_loss, auc_score


def inference(model, loader, device):
    """
    Generates predictions for the test set (or any loader without labels).

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Data loader (returns images, ids).
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (ids, probabilities)
            ids (np.ndarray): Array of recording IDs.
            probabilities (np.ndarray): Array of predicted probabilities (N, NumClasses).
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns (images, ids)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                if len(batch) > 1:
                    ids = batch[1]
                    all_ids.append(ids.numpy())
            else:
                images = batch

            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        final_probs = np.concatenate(all_probs, axis=0)
    else:
        final_probs = np.array([])

    if len(all_ids) > 0:
        final_ids = np.concatenate(all_ids, axis=0)
    else:
        final_ids = np.array([])

    return final_ids, final_probs
