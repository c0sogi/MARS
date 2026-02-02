import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import get_device


def train_one_epoch(model, loader, optimizer, device, epoch, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run training on.
        epoch (int): Current epoch number (for logging).
        max_batches (int, optional): Limit the number of batches for debugging.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    criterion = nn.BCEWithLogitsLoss()

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape [batch, 1]

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def validate_one_epoch(model, loader, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.
        max_batches (int, optional): Limit the number of batches for debugging.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    # Print full precision as requested
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def predict_with_tta(model, loader, device, max_batches=None):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'label' (probability).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i, (images, ids) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)

            # 1. Forward pass original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass flipped (TTA)
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (N, C, H, W)
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # Move to CPU
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            ids_np = ids.numpy().flatten()

            for img_id, prob in zip(ids_np, avg_probs_np):
                results.append({"id": int(img_id), "label": float(prob)})

    df = pd.DataFrame(results)
    return df


def save_submission(df, output_path="./submission/submission.csv"):
    """
    Saves the prediction DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'label' columns.
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
