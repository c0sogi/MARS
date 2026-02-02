import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
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
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of the Mean Absolute Error, calculated for each scalar
    coupling type, and then averaged across types.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        types (array-like): The scalar coupling type for each sample.

    Returns:
        float: The Log MAE metric.
    """
    # Create a DataFrame to handle grouping easily
    df_metric = pd.DataFrame(
        {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "type": np.array(types),
        }
    )

    # Calculate Absolute Error
    df_metric["mae"] = np.abs(df_metric["y_true"] - df_metric["y_pred"])

    # Calculate MAE per type
    mae_per_type = df_metric.groupby("type")["mae"].mean()

    # Calculate Log of MAE
    # We use natural log (np.log) as is standard for this metric description
    log_mae_per_type = np.log(mae_per_type)

    # Average across types
    score = log_mae_per_type.mean()

    return score
