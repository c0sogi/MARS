import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (cuda or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_mcrmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    according to the competition metric.

    Logic:
    1. Slices predictions to the first `seq_scored` positions (68).
    2. Selects only the scored columns defined in Config.SCORED_COLS.
    3. Computes RMSE for each selected column.
    4. Returns the mean of these RMSE values.

    Args:
        preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, 5).
                              Usually Seq_Len is 107.
        targets (torch.Tensor): Ground truth targets of shape (Batch, Seq_Scored, 5).
                                Usually Seq_Scored is 68.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are on the same device and are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Move to CPU for metric calculation to avoid synchronization overhead/issues if mixed
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. Slice predictions to match the scored sequence length (68)
    # targets are already length 68 based on data loading, preds are 107
    seq_scored = Config.PRED_LEN
    preds_sliced = preds[:, :seq_scored, :]

    # Verify shapes match in the sequence dimension
    if preds_sliced.shape[1] != targets.shape[1]:
        raise ValueError(
            f"Shape mismatch after slicing: Preds {preds_sliced.shape}, Targets {targets.shape}. "
            f"Expected sequence length {seq_scored}."
        )

    # 2. Identify indices of the scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # 3. Filter tensors to only scored columns
    # Shape becomes (Batch, Seq_Scored, 3)
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # 4. Compute MSE per column
    # (y - y_hat)^2
    mse = (targets_filtered - preds_filtered) ** 2

    # Average over batch and sequence length dimensions (0 and 1), keeping column dim (2)
    # Result shape: (3,)
    mse_per_col = torch.mean(mse, dim=(0, 1))

    # 5. Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # 6. Compute Mean of RMSEs
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse.item()
