import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the mean of the RMSEs of each column.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Shape should be (N, L, 3) or (N*L, 3).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Shape should be (N, L, 3) or (N*L, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in MCRMSE: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate MSE per column (last dimension)
    # We average over all dimensions except the last one (columns)
    # If shape is (Batch, Seq, Channels), axis=(0, 1)
    # If shape is (Batch*Seq, Channels), axis=0
    axes = tuple(range(y_true.ndim - 1))
    mse_per_column = np.mean((y_true - y_pred) ** 2, axis=axes)

    # RMSE per column
    rmse_per_column = np.sqrt(mse_per_column)

    # Mean of RMSEs across columns
    score = np.mean(rmse_per_column)

    return float(score)


def create_submission_dataframe(ids, preds):
    """
    Converts raw model predictions into the competition submission format.

    Args:
        ids (list or np.ndarray): List of sample IDs.
        preds (np.ndarray or torch.Tensor): Predictions of shape (N_samples, 107, 3).
                                            Channels: reactivity, deg_Mg_pH10, deg_Mg_50C.

    Returns:
        pd.DataFrame: Formatted submission dataframe.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    N, L, C = preds.shape

    # Validation of shapes based on Config
    if L != Config.SEQ_LEN:
        raise ValueError(
            f"Prediction sequence length must be {Config.SEQ_LEN}, got {L}"
        )
    if C != 3:
        raise ValueError(f"Prediction channels must be 3, got {C}")
    if len(ids) != N:
        raise ValueError(f"Number of IDs ({len(ids)}) does not match predictions ({N})")

    # Flatten predictions: (N * 107, 3)
    preds_flat = preds.reshape(-1, 3)

    # Generate ID_seqpos keys
    # Repeat IDs: [id1, id1... id2, id2...]
    ids_repeated = np.repeat(ids, L)
    # Tile positions: [0, 1, ... 106, 0, 1, ... 106]
    seq_pos = np.tile(np.arange(L), N)

    # Construct id_seqpos strings
    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, seq_pos)]

    # Construct DataFrame
    # Model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # Submission requires: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # We fill the unscored columns (deg_pH10, deg_50C) with 0.0

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

    # Ensure column order matches the sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    return submission_df[cols]


def save_submission(ids, preds, save_path=Config.SUBMISSION_PATH):
    """
    Wraps creation and saving of the submission file.

    Args:
        ids (list): Sample IDs.
        preds (np.ndarray): Predictions (N, 107, 3).
        save_path (str): Output path.
    """
    df = create_submission_dataframe(ids, preds)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    # print(f"Submission saved to {save_path}") # Optional logging
