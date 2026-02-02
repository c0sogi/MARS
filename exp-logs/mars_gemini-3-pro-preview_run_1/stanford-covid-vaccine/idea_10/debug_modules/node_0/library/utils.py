import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the 3 scored targets.

    The metric is defined as the average of the RMSEs calculated for each of the
    scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Args:
        y_true (np.array): Ground truth values of shape (N_samples, 3).
        y_pred (np.array): Predicted values of shape (N_samples, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate Mean Squared Error for each column (axis 0 is samples)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse = np.sqrt(mse)

    # Calculate the Mean of the RMSEs across the 3 columns
    score = np.mean(rmse)

    return float(score)


def build_submission_df(ids, preds, config=Config()):
    """
    Formats the predictions into a submission DataFrame required for the competition.

    This function takes the predictions for the 3 scored columns, flattens them,
    generates the required 'id_seqpos' identifiers, and fills the unscored columns
    (deg_pH10, deg_50C) with zeros.

    Args:
        ids (list or np.array): List of sample IDs (length N).
        preds (np.array): Predictions of shape (N, seq_len, 3).
                          The 3 channels correspond to [reactivity, deg_Mg_pH10, deg_Mg_50C].
        config (Config): Configuration object containing sequence length.

    Returns:
        pd.DataFrame: A DataFrame with columns ['id_seqpos', 'reactivity', 'deg_Mg_pH10',
                      'deg_pH10', 'deg_Mg_50C', 'deg_50C'].
    """
    ids = np.array(ids)
    n_samples = len(ids)
    seq_len = config.SEQ_LEN

    # Validate input shapes
    if preds.shape[0] != n_samples:
        raise ValueError(
            f"Number of predictions ({preds.shape[0]}) does not match number of IDs ({n_samples})."
        )
    if preds.shape[1] != seq_len:
        raise ValueError(
            f"Prediction sequence length ({preds.shape[1]}) does not match config ({seq_len})."
        )
    if preds.shape[2] != 3:
        raise ValueError(
            f"Prediction channels ({preds.shape[2]}) do not match expected scored columns (3)."
        )

    # 1. Generate 'id_seqpos' column
    # Repeat IDs: [id1, id1... (seq_len times), id2, id2...]
    repeated_ids = np.repeat(ids, seq_len)

    # Tile positions: [0, 1, ... 106, 0, 1, ... 106]
    tiled_pos = np.tile(np.arange(seq_len), n_samples)

    # Combine to form strings like "id_001_0", "id_001_1", ...
    id_seqpos = [f"{i}_{p}" for i, p in zip(repeated_ids, tiled_pos)]

    # 2. Flatten predictions to (N_samples * seq_len, 3)
    flat_preds = preds.reshape(-1, 3)

    # 3. Construct DataFrame
    # Note: deg_pH10 and deg_50C are unscored and should be 0.0
    df = pd.DataFrame(
        {
            "id_seqpos": id_seqpos,
            "reactivity": flat_preds[:, 0],
            "deg_Mg_pH10": flat_preds[:, 1],
            "deg_pH10": 0.0,
            "deg_Mg_50C": flat_preds[:, 2],
            "deg_50C": 0.0,
        }
    )

    return df
