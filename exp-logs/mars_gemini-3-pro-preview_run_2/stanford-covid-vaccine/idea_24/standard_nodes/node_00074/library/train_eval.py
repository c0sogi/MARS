import torch
import numpy as np
from library.config import Config


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        criterion (torch.nn.Module): Loss function (MCRMSELoss).
        device (torch.device): Device to run training on (CPU/GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to the appropriate device
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects inputs and partner_indices
        outputs = model(inputs, partner_indices)

        # Compute loss
        # Criterion (MCRMSELoss) handles slicing and column filtering internally
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        num_batches += 1

    # Return average loss
    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the MCRMSE metric.

    Crucially, this function accumulates Sum of Squared Errors (SSE) and counts
    across the entire dataset before computing the final RMSE. This avoids the
    statistical bias introduced by averaging RMSE scores calculated per-batch.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: The global MCRMSE score.
    """
    model.eval()

    # Identify indices of the scored columns based on Config
    # Config.ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [Config.ALL_TARGETS.index(t) for t in Config.SCORED_TARGETS]

    # Create a tensor for index selection
    scored_indices_tensor = torch.tensor(scored_indices, device=device)

    # Accumulators for global RMSE calculation
    # We track SSE per column to compute column-wise RMSE later
    total_sse = torch.zeros(len(scored_indices), device=device)
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(inputs, partner_indices)

            # 1. Slice predictions to match target length (68)
            # Model outputs: (Batch, 107, 5) -> (Batch, 68, 5)
            preds_sliced = outputs[:, : Config.PRED_LEN, :]

            # 2. Select only the scored columns
            # Result shape: (Batch, 68, 3)
            preds_scored = torch.index_select(preds_sliced, 2, scored_indices_tensor)
            targets_scored = torch.index_select(targets, 2, scored_indices_tensor)

            # 3. Compute Squared Errors
            squared_diff = (preds_scored - targets_scored) ** 2

            # 4. Accumulate SSE per column
            # Sum over Batch (dim 0) and Sequence Length (dim 1)
            batch_sse = torch.sum(squared_diff, dim=(0, 1))
            total_sse += batch_sse

            # 5. Accumulate total count of elements per column
            # Batch Size * Sequence Length
            batch_count = targets_scored.shape[0] * targets_scored.shape[1]
            total_count += batch_count

    if total_count == 0:
        return 0.0

    # Compute Global MSE per column
    mse_per_column = total_sse / total_count

    # Compute Global RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Compute MCRMSE: Mean of the column-wise RMSEs
    mcrmse = torch.mean(rmse_per_column).item()

    return mcrmse
