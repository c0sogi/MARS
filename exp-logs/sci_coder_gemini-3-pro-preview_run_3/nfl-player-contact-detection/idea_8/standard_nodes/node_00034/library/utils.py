import os
import random
import numpy as np
import pandas as pd
import gc
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Torch not installed or not needed


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): Whether to print the memory reduction statistics.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(
                        np.float32
                    )  # float16 has low precision, safer to use float32
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def parse_contact_id(input_data, expand: bool = True):
    """
    Parses contact_id(s) into component parts: game_play, step, player1, player2.
    Format: game_key_play_id_step_player1_player2

    Args:
        input_data (str or pd.Series): The contact_id string or Series of strings.
        expand (bool): If True and input is Series, returns a DataFrame with columns.
                       If False, returns a Series of lists (or single list if input is str).

    Returns:
        pd.DataFrame, pd.Series, or list depending on input and expand parameter.
    """
    # Case 1: Single String
    if isinstance(input_data, str):
        parts = input_data.split("_")
        # parts: [game_key, play_id, step, player1, player2]
        game_play = f"{parts[0]}_{parts[1]}"
        step = int(parts[2])
        p1 = parts[3]
        p2 = parts[4]
        return {
            "game_play": game_play,
            "step": step,
            "nfl_player_id_1": p1,
            "nfl_player_id_2": p2,
        }

    # Case 2: Pandas Series
    elif isinstance(input_data, pd.Series):
        # Vectorized split
        # contact_id format: game_key_play_id_step_player1_player2
        splits = input_data.str.split("_", expand=True)

        if expand:
            df_out = pd.DataFrame(index=input_data.index)
            df_out["game_play"] = splits[0] + "_" + splits[1]
            df_out["step"] = splits[2].astype(int)
            df_out["nfl_player_id_1"] = splits[3]
            df_out["nfl_player_id_2"] = splits[4]
            return df_out
        else:
            return splits

    else:
        raise ValueError("Input must be a string or a pandas Series.")


def load_data(path: str, file_type: str = None, **kwargs) -> pd.DataFrame:
    """
    Safely loads data from a CSV or Parquet file.

    Args:
        path (str): Path to the file.
        file_type (str, optional): 'csv' or 'parquet'. If None, inferred from extension.
        **kwargs: Additional arguments passed to pandas read functions.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if file_type is None:
        if path.endswith(".parquet"):
            file_type = "parquet"
        elif path.endswith(".csv"):
            file_type = "csv"
        else:
            raise ValueError(
                "Cannot infer file type. Please specify 'csv' or 'parquet'."
            )

    if file_type == "csv":
        return pd.read_csv(path, **kwargs)
    elif file_type == "parquet":
        return pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
