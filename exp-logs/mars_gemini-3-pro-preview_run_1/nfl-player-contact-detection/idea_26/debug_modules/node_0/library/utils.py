import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
from scipy.ndimage import gaussian_filter1d
import library.config as config


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
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
                    df[col] = df[col].astype(np.float32)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(
        f"Memory usage of dataframe is {start_mem:.2f} MB --> {end_mem:.2f} MB (Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)"
    )
    return df


def calc_mcc(y_true, y_pred_proba, threshold=0.5):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        float: MCC score.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return matthews_corrcoef(y_true, y_pred)


def find_best_threshold(y_true, y_pred_proba, step=0.01):
    """
    Finds the threshold that maximizes MCC.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities.
        step: Step size for threshold search.

    Returns:
        float: Best threshold.
        float: Best MCC score.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Search range from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, step)

    for thresh in thresholds:
        score = calc_mcc(y_true, y_pred_proba, thresh)
        if score > best_mcc:
            best_mcc = score
            best_thresh = thresh

    return best_thresh, best_mcc


def gaussian_smooth_labels(df, sigma=config.SMOOTHING_SIGMA):
    """
    Applies Gaussian smoothing to the 'contact' labels over the temporal dimension (step)
    for each unique player pair within a play.

    Args:
        df: DataFrame containing ['game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'step', 'contact'].
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        df: DataFrame with an additional 'contact_smooth' column.
    """
    print(f"Applying Gaussian smoothing to labels with sigma={sigma}...")

    # Ensure sorted by grouping keys and time
    sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Create a unique group ID for each series
    # We combine game_play and player IDs.
    # Since nfl_player_id_2 can be 'G' (string) or int, convert to string for safe grouping combination
    group_keys = (
        df["game_play"].astype(str)
        + "_"
        + df["nfl_player_id_1"].astype(str)
        + "_"
        + df["nfl_player_id_2"].astype(str)
    )

    # Factorize to get integer group IDs
    group_ids, _ = pd.factorize(group_keys)

    # Extract data as numpy arrays for fast processing
    contact_values = df["contact"].values.astype(float)

    # Identify boundaries where group_id changes
    # np.diff(group_ids) != 0 gives indices where the next element is different
    # We prepend a 0 to align indices, but logic below works better with finding split points
    # Indices where the group changes:
    change_mask = np.diff(group_ids) != 0
    # Add 1 to indices because diff returns index i where arr[i+1] != arr[i]
    change_indices = np.where(change_mask)[0] + 1

    # Add start (0) and end (len) to indices
    split_indices = np.concatenate(([0], change_indices, [len(contact_values)]))

    # Result array
    smoothed_values = np.zeros_like(contact_values)

    # Iterate through slices
    # This loop runs once per player-pair-play combination
    # Using numpy slicing is significantly faster than pandas groupby apply
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]

        # Slice the contact array
        series = contact_values[start_idx:end_idx]

        # Apply filter
        # mode='nearest' extends the edge values, appropriate for contact duration
        smoothed_series = gaussian_filter1d(series, sigma=sigma, mode="nearest")

        smoothed_values[start_idx:end_idx] = smoothed_series

    # Clip to ensure valid probability range [0, 1] (though gaussian on 0/1 usually stays within bounds)
    smoothed_values = np.clip(smoothed_values, 0.0, 1.0)

    df["contact_smooth"] = smoothed_values

    return df
