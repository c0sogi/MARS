import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is computed as the mean of the RMSE values calculated for each
    target column independently.

    Formula: MCRMSE = (1/Nt) * sum_{j=1}^{Nt} sqrt( (1/n) * sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (N_samples, ..., N_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: matches y_true.

    Returns:
        float: The computed MCRMSE value.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in metric calculation: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Determine the number of targets (assumed to be the last dimension)
    num_targets = y_true.shape[-1]

    # Reshape inputs to (-1, num_targets) to treat all sequence positions/samples as a flat list per target
    y_true_flat = y_true.reshape(-1, num_targets)
    y_pred_flat = y_pred.reshape(-1, num_targets)

    # Calculate Mean Squared Error for each column
    mse_per_col = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
