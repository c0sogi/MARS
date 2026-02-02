import os
import random
import numpy as np
import pandas as pd
import hashlib
from library.config import SEED, STREAM_A_FEATURES, STREAM_B_FEATURES, LAG_OFFSETS


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and hash randomization.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # If torch is used later in the pipeline, it's good practice to seed it,
    # but strictly following requirements we focus on standard libs first.
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
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
                    )  # float16 has low precision, using float32 is safer
                elif (
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


def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Generates a MD5 hash based on the content of the pandas DataFrame.
    Used for cache invalidation.
    """
    # hash_pandas_object returns a series of hashes, one per row
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    # We hash the underlying bytes of the hash array
    full_hash = hashlib.md5(row_hashes.tobytes()).hexdigest()
    return full_hash


def verify_schema(df: pd.DataFrame, required_cols: list, name: str = "DataFrame"):
    """
    Validates that the DataFrame contains all required columns and that
    numeric feature columns are not zero-filled (which implies silent failure).

    Raises:
        RuntimeError: If validation fails.
    """
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"Schema Validation Failed for {name}: Missing columns {missing_cols}"
        )

    # Check for zero-filled columns (only for numeric types)
    # This prevents training on features that were defaulted to 0 due to merge errors
    zero_filled_cols = []
    for col in required_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if all values are exactly 0
            if (df[col] == 0).all():
                zero_filled_cols.append(col)

    if zero_filled_cols:
        raise RuntimeError(
            f"Schema Validation Failed for {name}: The following columns are zero-filled (all values are 0): {zero_filled_cols}. "
            "This indicates a pipeline error (e.g., failed merge or incorrect default imputation)."
        )

    print(f"Schema Validation Passed for {name}. Checked {len(required_cols)} columns.")


def compute_config_hash() -> str:
    """
    Computes a hash of the current configuration to ensure cache consistency.
    """
    # Create a string representation of the critical config components
    config_str = str(STREAM_A_FEATURES) + str(STREAM_B_FEATURES) + str(LAG_OFFSETS)
    return hashlib.md5(config_str.encode()).hexdigest()
