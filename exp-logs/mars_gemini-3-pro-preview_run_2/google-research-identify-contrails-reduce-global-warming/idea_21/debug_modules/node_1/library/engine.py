import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from library.config import DEVICE, SUBMISSION_PATH
from library.utils import rle_encode


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (nn.Module): Loss function (HybridLoss).
        device (str): Device to run training on ('cuda' or 'cpu').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in dataloader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Global Dice Coefficient.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run evaluation on.

    Returns:
        float: Global Dice Coefficient.
    """
    model.eval()

    intersection_sum = 0.0
    cardinality_sum = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Threshold predictions
            preds = (probs > 0.5).float()

            # Flatten for calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Accumulate stats for Global Dice
            # Intersection: |X n Y|
            intersection = (preds_flat * masks_flat).sum().item()
            # Cardinality: |X| + |Y|
            cardinality = preds_flat.sum().item() + masks_flat.sum().item()

            intersection_sum += intersection
            cardinality_sum += cardinality

    # Compute Global Dice: 2 * Intersection / Cardinality
    # Handle division by zero if dataset is completely empty (unlikely)
    if cardinality_sum == 0:
        return 1.0

    global_dice = (2.0 * intersection_sum) / cardinality_sum

    # Print full precision metric
    print(f"Validation Global Dice: {global_dice}")

    return global_dice


def predict_tta(model, images):
    """
    Performs Test-Time Augmentation (TTA) inference.
    Averages predictions from: Original, H-Flip, V-Flip, Rotate-180.

    Args:
        model (nn.Module): The neural network model.
        images (torch.Tensor): Batch of input images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability maps (B, C, H, W).
    """
    # 1. Original
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3 is width)
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)
    probs_h = torch.sigmoid(logits_h)
    probs_h = torch.flip(probs_h, dims=[3])  # Flip back

    # 3. Vertical Flip (dim 2 is height)
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)
    probs_v = torch.sigmoid(logits_v)
    probs_v = torch.flip(probs_v, dims=[2])  # Flip back

    # 4. Rotate 180 (H-Flip + V-Flip)
    images_rot = torch.flip(images, dims=[2, 3])
    logits_rot = model(images_rot)
    probs_rot = torch.sigmoid(logits_rot)
    probs_rot = torch.flip(probs_rot, dims=[2, 3])  # Flip back

    # Average probabilities
    avg_probs = (probs_orig + probs_h + probs_v + probs_rot) / 4.0

    return avg_probs


def inference(model, dataloader, device, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for test data (must be unshuffled).
        device (str): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()

    submission_data = []

    # Ensure we can access record_ids from the dataset
    # The dataloader must not be shuffled to map indices correctly
    dataset = dataloader.dataset

    # Track current index in the dataset
    current_idx = 0

    print("Starting inference with TTA...")

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Predict with TTA
            probs = predict_tta(model, images)

            # Threshold to binary mask
            preds = (probs > 0.5).float().cpu().numpy()

            # Process batch
            for i in range(batch_size):
                # Get record_id
                record_id = dataset.df.iloc[current_idx]["record_id"]

                # Get mask for this image (squeeze channel dim: 1, H, W -> H, W)
                mask = preds[i, 0, :, :]

                # Encode
                rle_str = rle_encode(mask)

                submission_data.append(
                    {"record_id": record_id, "encoded_pixels": rle_str}
                )

                current_idx += 1

    # Create DataFrame and save
    df_sub = pd.DataFrame(submission_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_sub)} records.")
