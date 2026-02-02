import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.data import mixup_data, mixup_criterion


def train_one_epoch(model, loader, criterion, optimizer, device, alpha=1.0):
    """
    Executes one epoch of training with Mixup regularization.

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        optimizer: The optimizer.
        device (torch.device): The device to run on.
        alpha (float): Mixup alpha parameter.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        # Ensure targets are (B, 1) for BCEWithLogitsLoss
        targets = targets.to(device).view(-1, 1)
        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute Mixup loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Training Loss: {epoch_loss:.10f}")
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion: The loss function.
        device (torch.device): The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case if only one class is present in the batch/set
        auc_score = 0.5

    print(f"Validation Loss: {epoch_loss:.10f} | AUC: {auc_score:.10f}")
    return epoch_loss, auc_score


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original + HFlip + VFlip).

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run on.

    Returns:
        dict: A dictionary mapping image_id (str) to predicted probability (float).
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for inputs, _, ids in loader:
            inputs = inputs.to(device)

            # 1. Original
            outputs_orig = model(inputs)
            probs_orig = torch.sigmoid(outputs_orig)

            # 2. Horizontal Flip (flip width dimension, usually dim 3 for NCHW)
            inputs_h = torch.flip(inputs, [3])
            outputs_h = model(inputs_h)
            probs_h = torch.sigmoid(outputs_h)

            # 3. Vertical Flip (flip height dimension, usually dim 2 for NCHW)
            inputs_v = torch.flip(inputs, [2])
            outputs_v = model(inputs_v)
            probs_v = torch.sigmoid(outputs_v)

            # Average the probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Convert to numpy array
            avg_probs_np = avg_probs.cpu().numpy().flatten()

            # Store predictions
            # ids is a tuple of strings from the dataloader
            for img_id, prob in zip(ids, avg_probs_np):
                predictions[img_id] = float(prob)

    return predictions
