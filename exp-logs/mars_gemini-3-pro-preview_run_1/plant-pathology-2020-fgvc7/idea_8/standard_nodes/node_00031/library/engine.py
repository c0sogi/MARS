import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel, update_bn

from library.config import Config
from library.utils import calculate_class_weights
from library.dataset import get_loaders
from library.model import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: Computation device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        # Targets from dataset are float32 (likely one-hot).
        # CrossEntropyLoss expects class indices (LongTensor) for 1D targets.
        target_indices = torch.argmax(targets, dim=1)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, target_indices)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def predict_and_submit(model):
    """
    Generates predictions using the model (with TTA) and saves the submission file.

    Args:
        model: The trained PyTorch model (SWA model).
    """
    _, test_loader = get_loaders(load_cached_data=True)

    model.eval()
    predictions = []

    # Extract image IDs from the dataset dataframe
    image_ids = test_loader.dataset.df["image_id"].tolist()

    print("Generating predictions with Test-Time Augmentation (Horizontal Flip)...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)

            # 1. Forward pass original
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass flipped (TTA)
            # Flip along width (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = torch.softmax(outputs_flipped, dim=1)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            predictions.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)

    # Create submission DataFrame
    df_sub = pd.DataFrame(predictions, columns=Config.CLASS_LABELS)
    df_sub.insert(0, "image_id", image_ids)

    # Save submission
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
