import os
import torch
import pandas as pd
import numpy as np
from library.utils import AverageMeter
from library.config import DEVICE, SUBMISSION_PATH


def train_one_epoch(model, loader, criterion, optimizer, device=DEVICE):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device=DEVICE):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (callable): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average loss on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg}")

    return losses.avg


def predict_with_tta(model, loader, device=DEVICE):
    """
    Generates predictions using Test-Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for test data (returns img, angle, id).
        device (torch.device): Device to run inference on.

    Returns:
        dict: Dictionary mapping image_id (str) to predicted probability (float).
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original Prediction
            out1 = model(images, angles)

            # 2. Horizontal Flip Prediction (dim 3 is width in NCHW)
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)

            # 3. Vertical Flip Prediction (dim 2 is height in NCHW)
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)

            # Average the predictions
            avg_preds = (out1 + out2 + out3) / 3.0

            # Convert to numpy array
            probs = avg_preds.cpu().numpy().flatten()

            # Store results
            for i, img_id in enumerate(ids):
                predictions[img_id] = float(probs[i])

    return predictions


def save_submission(predictions, output_path=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (dict): Dictionary mapping id to probability.
        output_path (str): Path to save the CSV file.
    """
    ids = []
    probs = []

    for img_id, prob in predictions.items():
        ids.append(img_id)
        probs.append(prob)

    df = pd.DataFrame({"id": ids, "is_iceberg": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
