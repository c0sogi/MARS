import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import MetricMonitor, mixup_data, mixup_criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup regularization.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (callable): Loss function (usually BCEWithLogitsLoss).
        optimizer (Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run computations on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Mixup alpha parameter from config
    alpha = Config.MIXUP_ALPHA

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        mixed_images, y_a, y_b, lam = mixup_data(images, targets, alpha, device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_images)

        # Squeeze outputs to match target shape (B, 1) -> (B)
        outputs = outputs.squeeze(1)

        # Compute loss using mixup criterion
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())

    print(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (callable): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update loss metric
            metric_monitor.update("Loss", loss.item())

            # Store predictions for AUC calculation
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate ROC AUC
    auc = roc_auc_score(all_targets, all_preds)
    metric_monitor.update("AUC", auc)

    # Print full precision metrics
    print(f"Validation: Loss: {metric_monitor.get_avg('Loss')}, AUC: {auc}")
    return metric_monitor.get_avg("Loss"), auc


def predict_with_tta(model, loader, device):
    """
    Performs inference using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run computations on.

    Returns:
        np.array: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # View 1: Original
            out1 = torch.sigmoid(model(images).squeeze(1))

            # View 2: Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h).squeeze(1))

            # View 3: Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v).squeeze(1))

            # View 4: Rotate 180 (Horizontal + Vertical Flip)
            images_hv = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(images_hv).squeeze(1))

            # Average predictions across all views
            avg_preds = (out1 + out2 + out3 + out4) / 4.0
            all_preds.extend(avg_preds.cpu().numpy())

    return np.array(all_preds)


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data (must be non-shuffled).
        device (torch.device): Device to run computations on.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission with TTA...")

    # Get predictions
    predictions = predict_with_tta(model, test_loader, device)

    # Load test metadata to retrieve IDs
    # We assume the loader preserves the order of the metadata file (shuffle=False)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Validation check
    if len(predictions) != len(test_meta):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match metadata ({len(test_meta)})"
        )

    # Create DataFrame
    submission = pd.DataFrame({"id": test_meta["id"], "has_cactus": predictions})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
