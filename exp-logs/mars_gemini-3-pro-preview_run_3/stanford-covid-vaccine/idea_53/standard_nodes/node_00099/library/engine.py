import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the average RMSE across specified target columns.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (N, Seq_Len, Num_Targets)
            targets: Tensor of shape (N, Seq_Len, Num_Targets)
        Returns:
            loss: Scalar tensor
        """
        # Calculate Squared Error
        mse = (preds - targets) ** 2

        # Average over Batch (dim 0) and Sequence (dim 1) to get MSE per column
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Average RMSE across columns
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer instance
        device: Device to run on
        scheduler: Learning rate scheduler (optional)

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    criterion = MCRMSELoss()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Move data to device
        sequence = batch["sequence"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)  # Shape: (N, 68, 5)

        optimizer.zero_grad()

        # Forward pass
        # Model output shape: (N, 107, 5)
        outputs = model(sequence, pair_indices)

        # Slice outputs to match target length (first 68 positions)
        outputs_sliced = outputs[:, : Config.PRED_LEN, :]

        # Calculate loss on ALL 5 targets (Multi-Task Learning)
        loss = criterion(outputs_sliced, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Crucial for deep RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        batch_size = sequence.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    # Step scheduler if it's epoch-based (CosineAnnealing is usually stepped per epoch)
    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / total_samples
    return avg_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes MCRMSE on the 3 scored columns.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: Device to run on

    Returns:
        score: MCRMSE score on the scored columns
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)  # Shape: (N, 68, 5)

            # Forward pass
            outputs = model(sequence, pair_indices)

            # Slice outputs to match target length
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            # Collect predictions and targets
            all_preds.append(outputs_sliced.cpu())
            all_targets.append(targets.cpu())

    # Aggregate globally
    all_preds = torch.cat(all_preds, dim=0)  # (Total_Samples, 68, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (Total_Samples, 68, 5)

    # Filter for Scored Columns only (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Indices: [0, 1, 3]
    scored_indices = Config.SCORED_INDICES

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Calculate MCRMSE manually for the scored subset
    mse = (preds_scored - targets_scored) ** 2
    mse_per_col = torch.mean(mse, dim=(0, 1))
    rmse_per_col = torch.sqrt(mse_per_col)
    score = torch.mean(rmse_per_col).item()

    return score
