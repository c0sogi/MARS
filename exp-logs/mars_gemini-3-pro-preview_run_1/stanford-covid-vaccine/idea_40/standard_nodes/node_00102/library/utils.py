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
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )
    where Nt is the number of target columns (3 for this task).

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
                                           Shape: (N, seq_len, num_targets) or (N*seq_len, num_targets).
        y_pred (np.array or torch.Tensor): Predicted values.
                                           Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # The last dimension represents the different targets (reactivity, deg_Mg_pH10, deg_Mg_50C)
    num_targets = y_true.shape[-1]
    rmses = []

    for i in range(num_targets):
        # Extract the specific column
        y_t_col = y_true[..., i]
        y_p_col = y_pred[..., i]

        # Flatten to compute MSE over all samples and positions for this column
        y_t_flat = y_t_col.flatten()
        y_p_flat = y_p_col.flatten()

        # Calculate MSE
        mse = np.mean((y_t_flat - y_p_flat) ** 2)

        # Calculate RMSE
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    # Return the mean of the column-wise RMSEs
    return np.mean(rmses)
