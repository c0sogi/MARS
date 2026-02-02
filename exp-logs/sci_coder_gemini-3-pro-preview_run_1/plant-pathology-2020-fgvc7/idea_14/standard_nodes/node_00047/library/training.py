import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_metric


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for metric calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metric using the provided utility
    score = calculate_metric(all_labels, all_preds)

    return avg_loss, score


def predict(model, test_loader, device, use_tta=False):
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model (torch.nn.Module): The model to use for inference.
        test_loader (DataLoader): The test data loader.
        device (torch.device): The device to use.
        use_tta (bool): Whether to use Test-Time Augmentation.

    Returns:
        tuple: (image_ids, predictions)
            image_ids (list): List of image IDs.
            predictions (np.ndarray): Array of predicted probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Base predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # Horizontal Flip
                images_h = torch.flip(images, [3])
                outputs_h = model(images_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # Vertical Flip
                images_v = torch.flip(images, [2])
                outputs_v = model(images_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average predictions
                # Weighting could be applied here, but simple average is robust
                probs = (probs + probs_h + probs_v) / 3.0

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(image_ids)

    predictions = np.concatenate(all_preds, axis=0)
    return all_ids, predictions
