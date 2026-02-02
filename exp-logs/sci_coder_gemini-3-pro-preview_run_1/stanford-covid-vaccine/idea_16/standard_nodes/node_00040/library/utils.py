import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Args:
        y_true: Ground truth values (N, seq_len, num_targets) or flattened.
        y_pred: Predicted values (N, seq_len, num_targets) or flattened.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten sample and sequence dimensions if inputs are 3D (N, L, C) -> (N*L, C)
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
    if y_pred.ndim == 3:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate Mean Squared Error for each target column
    # axis=0 averages over the samples/positions
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse = np.sqrt(mse)

    # Calculate the average RMSE across all target columns
    mcrmse = np.mean(rmse)

    return float(mcrmse)


def format_submission(ids, preds, save_path=Config.SUBMISSION_FILE_PATH):
    """
    Formats the predictions into the competition submission format.

    Args:
        ids (list): List of sample IDs.
        preds (np.ndarray): Array of shape (num_samples, seq_len, 3) containing predictions
                            for ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
        save_path (str): Path to save the CSV file.
    """
    # Ensure predictions are numpy array
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    num_samples, seq_len, num_preds = preds.shape

    # Verify shapes match
    if len(ids) != num_samples:
        raise ValueError(
            f"Number of IDs ({len(ids)}) does not match number of prediction samples ({num_samples})"
        )

    # Flatten the predictions to (num_samples * seq_len, num_preds)
    preds_flat = preds.reshape(-1, num_preds)

    # Create the id_seqpos column efficiently
    # Repeat each ID seq_len times
    ids_repeated = np.repeat(ids, seq_len)
    # Tile the sequence positions (0, 1, ..., seq_len-1) num_samples times
    seqpos_tiled = np.tile(np.arange(seq_len), num_samples)
    # Combine into strings
    id_seqpos = [f"{i}_{s}" for i, s in zip(ids_repeated, seqpos_tiled)]

    # Construct the DataFrame
    # The submission requires 5 columns. The model predicts 3.
    # Mapping:
    #   reactivity   -> preds[:, 0]
    #   deg_Mg_pH10  -> preds[:, 1]
    #   deg_pH10     -> 0.0 (Not scored/predicted)
    #   deg_Mg_50C   -> preds[:, 2]
    #   deg_50C      -> 0.0 (Not scored/predicted)

    submission_df = pd.DataFrame(
        {
            "id_seqpos": id_seqpos,
            "reactivity": preds_flat[:, 0],
            "deg_Mg_pH10": preds_flat[:, 1],
            "deg_pH10": 0.0,
            "deg_Mg_50C": preds_flat[:, 2],
            "deg_50C": 0.0,
        }
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
