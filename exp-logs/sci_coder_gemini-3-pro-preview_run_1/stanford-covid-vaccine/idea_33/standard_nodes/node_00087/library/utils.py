import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).
    This function calculates the RMSE for each target column separately and then
    averages them. It can be used as a validation metric or a loss function.

    Args:
        y_true (torch.Tensor): Ground truth values. Shape (N, 3) or (N, L, 3).
        y_pred (torch.Tensor): Predicted values. Shape (N, 3) or (N, L, 3).

    Returns:
        torch.Tensor: Scalar tensor containing the MCRMSE value.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Flatten to (N_total, 3) where 3 is the number of targets
    # We assume the last dimension corresponds to the targets (reactivity, deg_Mg_pH10, deg_Mg_50C)
    num_targets = y_true.shape[-1]
    y_true_flat = y_true.view(-1, num_targets)
    y_pred_flat = y_pred.view(-1, num_targets)

    # Compute MSE for each target column individually
    mse = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)

    # Compute RMSE for each target column
    rmse = torch.sqrt(mse)

    # Compute the mean of the RMSEs across the columns
    loss = torch.mean(rmse)

    return loss


def format_submission(ids, preds, seq_length=107):
    """
    Formats the predictions into a DataFrame suitable for submission.

    Args:
        ids (list): List of sample IDs (strings).
        preds (torch.Tensor or np.ndarray): Predictions of shape (N_samples, seq_length, 3).
            The 3 channels are expected to be [reactivity, deg_Mg_pH10, deg_Mg_50C].
        seq_length (int): The length of the sequence (default 107).

    Returns:
        pd.DataFrame: A DataFrame with columns ['id_seqpos', 'reactivity', 'deg_Mg_pH10',
                      'deg_pH10', 'deg_Mg_50C', 'deg_50C'].
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    n_samples = len(ids)

    # Basic shape validation
    if preds.shape[0] != n_samples:
        raise ValueError(
            f"Number of predictions ({preds.shape[0]}) does not match number of IDs ({n_samples})"
        )
    if preds.shape[1] != seq_length:
        raise ValueError(
            f"Prediction sequence length ({preds.shape[1]}) does not match expected ({seq_length})"
        )

    # Vectorized generation of id_seqpos
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(ids, seq_length)

    # Tile seqpos: [0, 1, ..., 106, 0, 1, ..., 106]
    seqpos_tiled = np.tile(np.arange(seq_length), n_samples)

    # Create id_seqpos strings
    # Using list comprehension as it's efficient for string formatting
    id_seqpos = [f"{i}_{s}" for i, s in zip(ids_repeated, seqpos_tiled)]

    # Flatten predictions to (N_samples * seq_length, 3)
    preds_flat = preds.reshape(-1, 3)

    # Create DataFrame
    # Mapping based on competition requirements:
    #   reactivity   -> preds[:, 0]
    #   deg_Mg_pH10  -> preds[:, 1]
    #   deg_pH10     -> 0.0 (Unscored, not predicted)
    #   deg_Mg_50C   -> preds[:, 2]
    #   deg_50C      -> 0.0 (Unscored, not predicted)

    df = pd.DataFrame(
        {
            "id_seqpos": id_seqpos,
            "reactivity": preds_flat[:, 0],
            "deg_Mg_pH10": preds_flat[:, 1],
            "deg_pH10": 0.0,
            "deg_Mg_50C": preds_flat[:, 2],
            "deg_50C": 0.0,
        }
    )

    return df
