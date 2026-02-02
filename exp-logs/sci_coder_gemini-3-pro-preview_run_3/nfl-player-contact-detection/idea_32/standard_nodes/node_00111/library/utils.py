import hashlib
import json
import pandas as pd
import numpy as np
from library.config import Config


def get_config_hash():
    """
    Generates a unique hash based on the current configuration state.
    This hash is used for cache invalidation. If critical parameters change
    (e.g., window size, lags, model params), the hash changes, forcing
    re-computation of features or models.

    Returns:
        str: A hexadecimal hash string representing the config state.
    """
    # Extract relevant configuration parameters that affect data processing and training
    config_state = {
        "SEED": Config.SEED,
        "WINDOW_SIZE": Config.WINDOW_SIZE,
        "VISUAL_CONSENSUS_LAGS": Config.VISUAL_CONSENSUS_LAGS,
        "UNDERSAMPLE_RATIO": Config.UNDERSAMPLE_RATIO,
        "XGB_PARAMS_STREAM_A": Config.XGB_PARAMS_STREAM_A,
        "XGB_PARAMS_STREAM_B": Config.XGB_PARAMS_STREAM_B,
    }

    # Serialize to JSON with sorted keys for determinism
    config_str = json.dumps(config_state, sort_keys=True)

    # Generate MD5 hash
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return config_hash


def validate_schema(
    df: pd.DataFrame, expected_columns: list, check_zero_filled: bool = True
):
    """
    Enforces strict schema validation on a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list): List of column names that must exist.
        check_zero_filled (bool): If True, raises error if any expected column
                                  contains only zeros (indicating potential feature engineering failure).

    Raises:
        RuntimeError: If columns are missing or (optionally) if columns are zero-filled.
    """
    # 1. Check for missing columns
    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        raise RuntimeError(
            f"Schema Validation Failed. The following {len(missing_cols)} columns are missing:\n"
            f"{missing_cols}"
        )

    # 2. Check for zero-filled columns (Pipeline Integrity)
    # This catches silent failures where features are created but not populated correctly.
    if check_zero_filled:
        zero_filled_cols = []
        for col in expected_columns:
            # We only check numeric columns for zero-filling
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if all values are 0 (or close to 0 for floats)
                if (df[col] == 0).all():
                    zero_filled_cols.append(col)

        if zero_filled_cols:
            raise RuntimeError(
                f"Schema Validation Failed. The following {len(zero_filled_cols)} columns are entirely zero-filled, "
                f"indicating a potential feature engineering error:\n{zero_filled_cols}"
            )

    return True
