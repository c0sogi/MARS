import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import mcrmse_loss


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Device to run training on.
        criterion (nn.Module): Loss function (expected MSELoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move data to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        y = batch["y"].to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(seq, loop, dist)

        # Masked Loss: Only calculate loss for the first 68 positions
        # preds shape: (Batch, Seq_Len, 3)
        # y shape: (Batch, Seq_Len, 3)
        preds_sliced = preds[:, : Config.pred_len, :]
        y_sliced = y[:, : Config.pred_len, :]

        loss = criterion(preds_sliced, y_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimization step
        optimizer.step()

        # Accumulate loss (weighted by batch size for accurate mean)
        batch_size = seq.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            y = batch["y"].to(device)

            # Forward pass
            preds = model(seq, loop, dist)

            # Slice to scored positions (first 68)
            preds_sliced = preds[:, : Config.pred_len, :]
            y_sliced = y[:, : Config.pred_len, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(y_sliced.cpu().numpy())

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE
        score = mcrmse_loss(y_true, y_pred)
    else:
        score = 0.0

    return score
