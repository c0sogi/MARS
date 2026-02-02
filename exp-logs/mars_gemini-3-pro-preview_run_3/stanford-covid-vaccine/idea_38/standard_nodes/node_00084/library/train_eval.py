import torch
import torch.nn as nn
from library.loss_metric import calculate_mcrmse


def train_epoch(model, loader, optimizer, criterion, config, device):
    """
    Executes one training epoch, including forward pass, loss calculation,
    backpropagation, and gradient clipping.

    Args:
        model (nn.Module): The RNA degradation prediction model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (nn.Module): Loss function (MCRMSELoss).
        config (Config): Configuration object containing hyperparameters.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        float: The average loss over the epoch.
    """
    model.train()
    train_loss_accum = 0.0
    num_batches = len(loader)

    for batch in loader:
        # Move inputs and targets to the configured device
        inputs = batch["sequence"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        bpp_mask = batch["bpp_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model expects: inputs, bpp_indices, bpp_mask
        preds = model(inputs, bpp_indices, bpp_mask)

        # Loss calculation
        # Criterion expects: preds (N, 107, 5), targets (N, 68, 5)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        # Mandatory for deep hybrid/RNN architectures to prevent exploding gradients
        if hasattr(config, "max_grad_norm") and config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        train_loss_accum += loss.item()

    # Calculate average loss
    avg_train_loss = train_loss_accum / num_batches if num_batches > 0 else 0.0
    return avg_train_loss


def validate(model, loader, config, device):
    """
    Evaluates the model on the validation set using the official competition metric.
    Aggregates predictions globally before calculating MCRMSE.

    Args:
        model (nn.Module): The RNA degradation prediction model.
        loader (DataLoader): DataLoader for the validation set.
        config (Config): Configuration object.
        device (torch.device): Computation device.

    Returns:
        float: The calculated MCRMSE score on the scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)

            # Targets are kept on CPU to avoid unnecessary GPU memory usage during aggregation
            targets = batch["targets"]

            # Forward pass
            preds = model(inputs, bpp_indices, bpp_mask)

            # Move predictions to CPU
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Concatenate all batches to form global tensors
    # Preds: (Total_Samples, 107, 5)
    # Targets: (Total_Samples, 68, 5)
    if len(all_preds) > 0:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
    else:
        return 0.0

    # Calculate MCRMSE
    # The calculate_mcrmse function handles:
    # 1. Slicing predictions to the first 68 positions.
    # 2. Filtering for the 3 scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    val_mcrmse = calculate_mcrmse(all_preds_tensor, all_targets_tensor)

    return val_mcrmse
