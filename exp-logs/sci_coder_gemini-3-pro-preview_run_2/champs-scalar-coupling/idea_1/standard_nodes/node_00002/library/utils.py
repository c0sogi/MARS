import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_mae(preds, targets, types) -> float:
    """
    Calculates the Log of the Mean Absolute Error (LogMAE) for each scalar coupling type,
    and then averages these values across all types.

    Metric = Mean( Log( MAE(type_i) ) ) for all types i.

    Args:
        preds: Predicted coupling constants. Can be a numpy array or torch.Tensor.
        targets: Actual coupling constants. Can be a numpy array or torch.Tensor.
        types: The coupling type for each prediction (e.g., '1JHC', 0).
               Can be a numpy array, torch.Tensor, or list.

    Returns:
        float: The calculated LogMAE metric.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(types, torch.Tensor):
        types = types.detach().cpu().numpy()

    # Ensure inputs are 1D arrays
    preds = np.squeeze(preds)
    targets = np.squeeze(targets)
    types = np.squeeze(types)

    # Verify shapes match
    if preds.shape != targets.shape or preds.shape != types.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape}, targets {targets.shape}, types {types.shape}"
        )

    # Create a DataFrame to facilitate grouping by coupling type
    df = pd.DataFrame({"pred": preds, "target": targets, "type": types})

    # Calculate absolute error for each prediction
    df["abs_error"] = np.abs(df["pred"] - df["target"])

    # Calculate Mean Absolute Error (MAE) for each coupling type
    mae_per_type = df.groupby("type")["abs_error"].mean()

    # Calculate the natural logarithm of the MAE for each type
    # Note: We assume MAE > 0. In physical datasets, exact 0.0 error is extremely rare.
    # If MAE is 0, log is -inf. We add a tiny epsilon for numerical stability if needed,
    # but strictly following the metric definition implies log(MAE).
    log_mae_per_type = np.log(mae_per_type + 1e-9)

    # Calculate the average of the Log MAEs across all types
    metric = log_mae_per_type.mean()

    return float(metric)
