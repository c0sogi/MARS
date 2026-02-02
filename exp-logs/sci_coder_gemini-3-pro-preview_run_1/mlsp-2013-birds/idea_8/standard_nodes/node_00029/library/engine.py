import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, swa_handler=None, swa_start_epoch=None
):
    """
    Executes one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): Training data loader.
        device (str): Device to use ('cpu' or 'cuda').
        epoch (int): Current epoch number (0-indexed).
        swa_handler (SWAHandler, optional): Handler for SWA updates.
        swa_start_epoch (int, optional): Epoch to start SWA updates.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Multi-label classification uses BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Determine Mixup application (Cite Lesson 7)
    # Disable Mixup during SWA to ensure stable convergence to the flat minimum
    apply_mixup = Config.USE_MIXUP
    if swa_start_epoch is not None and epoch >= swa_start_epoch:
        apply_mixup = False

    for batch_idx, (images, targets, _) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        if apply_mixup:
            # Mixup Implementation
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]

            outputs = model(mixed_images)

            # Mixup Loss (Linear combination of losses)
            loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(
                outputs, targets[index]
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Stochastic Weight Averaging (SWA) Update
    # SWA typically updates the averaged model at the end of each epoch
    if swa_handler is not None and swa_start_epoch is not None:
        if epoch >= swa_start_epoch:
            swa_handler.update(model)

    return epoch_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): Validation data loader.
        device (str): Device to use.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc_score = calculate_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    # Print metrics with full precision as requested
    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation AUC: {auc_score}")

    return epoch_loss, auc_score


def predict(model, data_loader, device):
    """
    Generates predictions for a dataset.

    Args:
        model (torch.nn.Module): The model to use.
        data_loader (torch.utils.data.DataLoader): Data loader.
        device (str): Device to use.

    Returns:
        tuple: (predictions, ids)
            predictions (np.ndarray): Probability matrix (N, NumClasses).
            ids (np.ndarray): Array of recording IDs (N,).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in data_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(ids.numpy())

    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds)
        ids = np.concatenate(all_ids)
    else:
        predictions = np.array([])
        ids = np.array([])

    return predictions, ids
