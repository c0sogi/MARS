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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mcrmse(preds, targets):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the competition.

    The metric is calculated by:
    1. Slicing predictions and targets to the first `seq_scored` positions (68).
    2. Filtering for the specific scoring columns: 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'.
    3. Calculating RMSE for each column independently.
    4. Taking the mean of the column-wise RMSEs.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (N, Seq_Len, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (N, Seq_Len, 5).

    Returns:
        float: The computed MCRMSE score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure tensors are on CPU for calculation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # Slice to the scored sequence length (68)
    seq_scored = Config.SEQ_SCORED

    # Handle slicing if dimensions exceed seq_scored
    if preds.shape[1] > seq_scored:
        preds = preds[:, :seq_scored, :]

    if targets.shape[1] > seq_scored:
        targets = targets[:, :seq_scored, :]

    # Identify indices of the columns used for scoring
    # TARGET_COLS order corresponds to the channel dimension of the tensors
    target_cols = Config.TARGET_COLS
    scoring_cols = Config.SCORING_COLS

    # Get indices for [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # Typically indices [0, 1, 3] based on the standard config
    scoring_indices = [i for i, col in enumerate(target_cols) if col in scoring_cols]

    # Filter predictions and targets to only the scoring columns
    preds_filtered = preds[:, :, scoring_indices]
    targets_filtered = targets[:, :, scoring_indices]

    # Calculate RMSE per column
    # Flatten batch and sequence dimensions to calculate global RMSE per column
    # Shape becomes (N * seq_scored, num_scoring_cols)
    preds_flat = preds_filtered.reshape(-1, len(scoring_indices))
    targets_flat = targets_filtered.reshape(-1, len(scoring_indices))

    # Mean Squared Error per column
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)

    # Root Mean Squared Error per column
    rmse = torch.sqrt(mse)

    # MCRMSE is the mean of the RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
