import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (str): The device to use for training (cpu/cuda).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples

    # Print metric with full precision
    print(f"Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate_tta(model, dataloader, device):
    """
    Performs validation using Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, and Vertical Flip.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Validation dataloader (returns image, label).
        device (str): The device to use.

    Returns:
        tuple: (probabilities, targets)
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # 1. Original Image
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            probs_h = F.softmax(outputs_h, dim=1)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            probs_v = F.softmax(outputs_v, dim=1)

            # Average
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_probs.append(avg_probs.cpu().numpy())
            all_targets.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_targets)


def predict_tta(model, dataloader, device):
    """
    Performs inference on the test set using Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, and Vertical Flip.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Test dataloader (returns image, image_id).
        device (str): The device to use for inference.

    Returns:
        pd.DataFrame: DataFrame containing image_ids and predicted probabilities.
    """
    model.eval()

    all_image_ids = []
    all_probs = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            # 1. Original Image
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            probs_h = F.softmax(outputs_h, dim=1)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            probs_v = F.softmax(outputs_v, dim=1)

            # Average the probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_probs.append(avg_probs.cpu().numpy())
            all_image_ids.extend(image_ids)

    # Concatenate all batch results
    predictions = np.concatenate(all_probs, axis=0)

    # Create DataFrame
    df = pd.DataFrame(predictions, columns=Config.CLASS_LABELS)
    df.insert(0, "image_id", all_image_ids)

    return df


def save_submission(df, path=Config.SUBMISSION_PATH):
    """
    Saves the prediction DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing predictions.
        path (str): The file path to save the submission.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save to CSV without the index
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
