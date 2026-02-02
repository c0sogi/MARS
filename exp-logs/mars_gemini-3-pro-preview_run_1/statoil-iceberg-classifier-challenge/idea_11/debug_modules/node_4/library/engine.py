import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library import config


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network.
        dataloader (DataLoader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (str): Device to run training on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        # Unpack batch (handle cases with/without labels if necessary, though train usually has labels)
        if len(batch) == 4:
            images, angles, labels, _ = batch
        else:
            raise ValueError("Training dataloader must provide labels.")

        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a validation set.

    Args:
        model (torch.nn.Module): The neural network.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to run evaluation on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, angles, labels, _ = batch
            else:
                raise ValueError("Validation dataloader must provide labels.")

            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    print(f"Validation Loss: {avg_loss}")

    return avg_loss


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Original + H-Flip + V-Flip).

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): Test dataloader.
        device (str): Device to run inference on.

    Returns:
        tuple: (ids, probabilities)
            ids (list): List of image IDs.
            probabilities (list): List of predicted probabilities (0-1).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle dataloader yielding 3 items (test) or 4 items (val)
            if len(batch) == 3:
                images, angles, ids = batch
            elif len(batch) == 4:
                images, angles, _, ids = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            images = images.to(device)
            angles = angles.to(device)

            # 1. Original View
            out_orig = model(images, angles)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (Width is dim 3: B, C, H, W)
            images_h = torch.flip(images, [3])
            out_h = model(images_h, angles)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (Height is dim 2: B, C, H, W)
            images_v = torch.flip(images, [2])
            out_v = model(images_v, angles)
            prob_v = torch.sigmoid(out_v)

            # Average the probabilities
            prob_avg = (prob_orig + prob_h + prob_v) / 3.0

            # Collect results
            all_preds.extend(prob_avg.cpu().numpy().flatten().tolist())
            all_ids.extend(ids)

    return all_ids, all_preds


def save_submission(ids, probabilities, filename=config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list): List of image IDs.
        probabilities (list): List of predicted probabilities.
        filename (str): Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame({"id": ids, "is_iceberg": probabilities})

    # Format matches sample_submission.csv
    df.to_csv(filename, index=False, float_format="%.6f")
    print(f"Submission saved to {filename}")
