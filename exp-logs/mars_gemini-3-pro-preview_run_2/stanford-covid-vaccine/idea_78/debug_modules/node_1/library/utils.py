import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_global_mcrmse(model, loader, device, scored_indices=None, pred_len=None):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) over the entire dataset.

    This function accumulates the Sum of Squared Errors (SSE) globally across all batches
    before calculating the square root. This avoids the bias introduced by averaging
    batch-level RMSEs.

    Args:
        model (nn.Module): The trained model to evaluate.
        loader (DataLoader): DataLoader containing validation/test data.
        device (torch.device): Device to perform computations on.
        scored_indices (list, optional): List of column indices to score.
                                         Defaults to Config.SCORED_INDICES.
        pred_len (int, optional): The length of the sequence to score.
                                  Defaults to Config.PRED_LEN.

    Returns:
        float: The global MCRMSE score.
    """
    # Use defaults from Config if not provided
    if scored_indices is None:
        scored_indices = Config.SCORED_INDICES
    if pred_len is None:
        pred_len = Config.PRED_LEN

    model.eval()

    # Initialize accumulators
    # We track SSE per column to compute column-wise RMSE later
    num_scored_cols = len(scored_indices)
    total_sse = torch.zeros(num_scored_cols, device=device)
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            # Unpack batch. Expecting (X, PartnerIndices, Y) for validation
            if len(batch) >= 3:
                x, p_idx, y = batch[0], batch[1], batch[2]
                y = y.to(device)
            else:
                raise ValueError(
                    "DataLoader must provide targets (Y) for metric calculation."
                )

            x = x.to(device)
            p_idx = p_idx.to(device)

            # Forward pass
            # The HC_SDRN model returns a tuple (y1, y2). We use y2 (the refined output).
            output = model(x, p_idx)
            if isinstance(output, tuple):
                _, y_pred = output
            else:
                y_pred = output

            # Slice predictions and targets to the valid scoring region
            # Shapes: (Batch, Seq_Len, 5) -> (Batch, Pred_Len, Num_Scored)
            y_pred_scored = y_pred[:, :pred_len, scored_indices]
            y_true_scored = y[:, :pred_len, scored_indices]

            # Calculate squared errors
            diff = y_pred_scored - y_true_scored
            squared_diff = diff**2

            # Sum SSE per column across batch and sequence length dimensions
            batch_sse = torch.sum(squared_diff, dim=(0, 1))

            # Accumulate global stats
            total_sse += batch_sse
            # Total count is Batch_Size * Pred_Len
            total_count += y_pred_scored.shape[0] * y_pred_scored.shape[1]

    # Compute global metrics
    # 1. Mean Squared Error per column
    mse = total_sse / total_count

    # 2. Root Mean Squared Error per column
    rmse = torch.sqrt(mse)

    # 3. Mean across columns (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
