import torch
import torch.nn as nn
from library.config import Config
from library.loss import MCRMSELoss


def get_scored_indices():
    """
    Identifies the indices of the columns used for scoring (reactivity, deg_Mg_pH10, deg_Mg_50C)
    within the full list of target columns.
    """
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_TARGET_COLS
    indices = [i for i, col in enumerate(target_cols) if col in scored_cols]
    return indices


def train_fn(model, data_loader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Strategy Compliance:
    - Multi-Task Learning: Optimizes on all 5 targets.
    - Gradient Clipping: Enforces max_norm=1.0 for stability.
    """
    model.train()
    total_loss = 0.0
    criterion = MCRMSELoss()

    for batch in data_loader:
        # Move data to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices)

        # Compute loss on ALL 5 targets (column_indices=None)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for MC-SD-BiGRU stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Strategy Compliance:
    - Global Aggregation: Concatenates all predictions before metric calculation.
    - Column Filtering: Calculates MCRMSE only on the 3 scored columns.
    """
    model.eval()

    # Lists to store global predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features, pair_indices)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate for global calculation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Determine indices for the 3 scored columns
    scored_indices = get_scored_indices()

    # Calculate MCRMSE specifically on scored columns
    criterion = MCRMSELoss()
    metric = criterion(all_preds, all_targets, column_indices=scored_indices)

    return metric.item()
