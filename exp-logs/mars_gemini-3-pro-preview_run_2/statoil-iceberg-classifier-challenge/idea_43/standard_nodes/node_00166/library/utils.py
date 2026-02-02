import os
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed=42):
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


def get_global_stats(json_path):
    """
    Iterates through the entire dataset at the specified path to calculate
    the global minimum and maximum values for Band 1 and Band 2.

    This function is designed to support global normalization strategies by
    providing dataset-wide statistics.

    Args:
        json_path (str): Path to the .json file containing the dataset
                         (e.g., 'input/train.json').

    Returns:
        dict: A dictionary containing the statistics with the following structure:
              {
                  "band_1": {"min": float, "max": float},
                  "band_2": {"min": float, "max": float}
              }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")

    # Load the data using pandas for efficiency
    # The file is expected to be a JSON array of objects
    df = pd.read_json(json_path)

    # Validate structure
    if "band_1" not in df.columns or "band_2" not in df.columns:
        raise ValueError(
            f"The file {json_path} does not contain required 'band_1' and 'band_2' columns."
        )

    # Flatten the lists of pixel values from all images into a single array per band
    # Each row in 'band_1'/'band_2' is a list of 5625 floats
    band_1_all = np.concatenate(df["band_1"].values)
    band_2_all = np.concatenate(df["band_2"].values)

    stats = {
        "band_1": {"min": float(np.min(band_1_all)), "max": float(np.max(band_1_all))},
        "band_2": {"min": float(np.min(band_2_all)), "max": float(np.max(band_2_all))},
    }

    return stats
