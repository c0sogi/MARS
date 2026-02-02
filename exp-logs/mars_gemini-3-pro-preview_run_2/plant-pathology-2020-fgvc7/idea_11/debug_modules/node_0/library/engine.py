import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def calculate_pos_weights(df, device):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class frequencies.
    Weight = (Total - Positive) / Positive

    Args:
        df (pd.DataFrame): The training dataframe containing target columns.
        device (torch.device): The device to store the weights on.

    Returns:
        torch.Tensor: Tensor of shape (num_classes,) containing positive weights.
    """
    # Extract target columns based on Config
    # Config.TARGET_COLUMNS are ['rust', 'scab'] for the decomposed task
    # The dataframe must have 'target_rust' and 'target_scab' as created in dataset.py

    target_cols = [f"target_{col}" for col in Config.TARGET_COLUMNS]

    # Ensure columns exist
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column {col} not found in dataframe for weight calculation."
            )

    targets = df[target_cols].values

    # Calculate counts
    pos_counts = np.sum(targets, axis=0)
    total_counts = len(df)

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)

    # Calculate weights: number of negatives / number of positives
    neg_counts = total_counts - pos_counts
    weights = neg_counts / pos_counts

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, optimizer, data_loader, device, epoch, pos_weights=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (DataLoader): The training data loader.
        device (torch.device): The device to run on.
        epoch (int): Current epoch number.
        pos_weights (torch.Tensor, optional): Positive class weights for BCE loss.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # Define Loss Function
    # We use BCEWithLogitsLoss which combines Sigmoid and BCE
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, data_loader, device, pos_weights=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run on.
        pos_weights (torch.Tensor, optional): Positive class weights for BCE loss (for consistency).

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)

            # Calculate loss
            loss = criterion(logits, targets)
            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store predictions and targets for metric calculation
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate ROC AUC
        # We use the utility function which handles macro averaging
        auc_score = calculate_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return avg_loss, auc_score
