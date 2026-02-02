import os
import torch
import numpy as np
import pandas as pd
from library.config import DEVICE, SUBMISSION_PATH
from library.utils import AverageMeter, get_roc_auc_score


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cpu or cuda).
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Ensure targets have the same shape as outputs (B, 1)
        targets = targets.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # Step the scheduler if provided (e.g., OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), inputs.size(0))

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Prints full precision metrics.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    loss_meter = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Ensure targets have the same shape as outputs (B, 1)
            targets = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))

            # Apply sigmoid to get probabilities from logits
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate results from all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
    else:
        all_targets = np.array([])
        all_preds = np.array([])

    # Flatten arrays for AUC calculation
    flat_targets = all_targets.ravel()
    flat_preds = all_preds.ravel()

    # Calculate AUC
    auc_score = get_roc_auc_score(flat_targets, flat_preds)

    # Print full precision metrics
    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation AUC: {auc_score}")

    return loss_meter.avg, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the dataset.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for inference.
        device: Device to run inference on.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
    else:
        all_preds = np.array([])

    return all_preds.ravel()


def generate_submission(model, dataloader, device, save_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for test data.
        device: Device to run inference on.
        save_path: Path to save the submission CSV.
    """
    print("Generating submission...")
    preds = predict(model, dataloader, device)

    # Retrieve IDs from the dataset
    # We assume dataloader.dataset has an 'ids' attribute as per library/dataset.py
    ids = dataloader.dataset.ids

    if len(ids) != len(preds):
        raise ValueError(
            f"Mismatch between number of IDs ({len(ids)}) and predictions ({len(preds)})"
        )

    submission_df = pd.DataFrame({"id": ids, "target": preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
