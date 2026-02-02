import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    return matthews_corrcoef(y_true, y_pred)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): If True, prints the memory reduction statistics.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Downcast float64 to float32.
                # float16 is often too low precision for physics calculations, so we stick to float32.
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        print(f"Memory usage after optimization is {end_mem:.2f} MB")
        print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df
