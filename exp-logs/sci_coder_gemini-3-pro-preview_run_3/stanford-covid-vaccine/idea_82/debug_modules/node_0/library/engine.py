import torch
import numpy as np
from library.config import Config


def train_fn(model, data_loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run on (cuda/cpu).
        max_grad_norm: Maximum norm for gradient clipping.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        inputs = batch["inputs"].to(device)
        neighbor_indices = batch["neighbor_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, neighbor_indices, pair_masks)

        # Slice to scored length (first 68 positions) for loss calculation
        # This ensures we don't train on the zero-padded tails
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]
        targets_sliced = targets[:, : Config.SEQ_SCORED, :]

        # Calculate loss on all 5 columns (as per strategy for multi-task learning)
        loss = criterion(preds_sliced, targets_sliced)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Metric: MCRMSE on 3 scored columns and first 68 positions.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for validation data.
        device: Device to run on.

    Returns:
        float: MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            neighbor_indices = batch["neighbor_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, neighbor_indices, pair_masks)

            # Slice to scored length (68)
            preds = preds[:, : Config.SEQ_SCORED, :]
            targets = targets[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Global aggregation
    if len(all_preds) == 0:
        return 0.0

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Filter for scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    scored_indices = [0, 1, 3]

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Flatten for MCRMSE calculation
    # Reshape to (N_total_positions, N_scored_columns)
    preds_flat = preds_scored.reshape(-1, len(scored_indices))
    targets_flat = targets_scored.reshape(-1, len(scored_indices))

    # Calculate MCRMSE
    # Mean Squared Error per column
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)
    # Root Mean Squared Error per column
    rmse = torch.sqrt(mse)
    # Mean of RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


def inference_fn(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for test data.
        device: Device to run on.

    Returns:
        tuple: (predictions numpy array, list of IDs)
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            neighbor_indices = batch["neighbor_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            # Predict on full length (107) as required for submission
            preds = model(inputs, neighbor_indices, pair_masks)

            preds_list.append(preds.cpu().numpy())
            ids_list.extend(ids)

    if len(preds_list) == 0:
        return np.array([]), []

    return np.concatenate(preds_list, axis=0), ids_list
