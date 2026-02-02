import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from library.utils import get_logger, calculate_score

# Initialize logger
logger = get_logger("Engine")


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Computation device.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (features, targets, mask, info) in enumerate(dataloader):
        # Move data to device
        features = features.to(device)  # (B, C, L)
        targets = targets.to(device)  # (B, 2, L)
        mask = mask.to(device)  # (B, L)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)  # (B, 2, L)

        # Calculate Loss (Mean Absolute Error)
        # We only calculate loss on valid time steps (mask == 1)
        loss_raw = torch.abs(outputs - targets)  # (B, 2, L)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, L)

        # Sum loss over valid elements and normalize
        # Add epsilon to denominator to avoid division by zero in case of empty sequence (unlikely)
        loss = (loss_raw * mask_expanded).sum() / (mask_expanded.sum() * 2 + 1e-8)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes the average MAE loss and the competition metric.

    The competition metric is calculated by:
    1. Computing distance errors for every valid timestamp.
    2. Calculating the mean of 50th and 95th percentile errors for each phone.
    3. Averaging these scores across all phones.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Computation device.

    Returns:
        tuple: (Average MAE Loss, Competition Score)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    # Dictionary to store errors per phone for metric calculation
    phone_errors = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (features, targets, mask, info) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            outputs = model(features)

            # Calculate Validation Loss (MAE)
            loss_raw = torch.abs(outputs - targets)
            mask_expanded = mask.unsqueeze(1)
            loss = (loss_raw * mask_expanded).sum() / (mask_expanded.sum() * 2 + 1e-8)

            running_loss += loss.item()
            count += 1

            # --- Metric Calculation Data Collection ---
            # Calculate Euclidean distance error in meters for each point
            # outputs: (B, 2, L) -> (DeltaEast, DeltaNorth)
            deltas = outputs - targets
            dist_errors = torch.sqrt(torch.sum(deltas**2, dim=1))  # (B, L)

            # Iterate through batch to collect errors per phone
            batch_size = features.size(0)
            phone_names = info["phone_name"]  # List of strings

            for i in range(batch_size):
                p_name = phone_names[i]

                # Extract valid errors using the mask
                # mask[i] is (L,)
                valid_mask = mask[i] > 0.5
                valid_errors = dist_errors[i][valid_mask].cpu().numpy()

                if len(valid_errors) > 0:
                    phone_errors[p_name].append(valid_errors)

    # Compute Competition Metric
    phone_scores = []
    for p_name, error_list in phone_errors.items():
        # Concatenate all errors for this phone across all batches/sequences
        if error_list:
            all_p_errors = np.concatenate(error_list)
            # calculate_score computes mean(p50, p95)
            s = calculate_score(all_p_errors)
            phone_scores.append(s)

    final_metric = np.mean(phone_scores) if phone_scores else 0.0
    avg_loss = running_loss / count if count > 0 else 0.0

    return avg_loss, final_metric
