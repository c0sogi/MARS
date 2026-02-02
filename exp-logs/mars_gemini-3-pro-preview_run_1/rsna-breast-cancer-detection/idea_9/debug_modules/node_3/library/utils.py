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
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_contralateral_lookup(df):
    """
    Creates a dictionary mapping each image_id to the file_path of its contralateral pair.

    A contralateral pair is defined as an image from the same patient and same view (e.g., CC, MLO),
    but with the opposite laterality (Left vs Right). This pre-computed lookup allows for O(1)
    retrieval during data loading, avoiding expensive dataframe filtering operations.

    Args:
        df (pd.DataFrame): Dataframe containing 'patient_id', 'view', 'laterality', 'image_id', and 'file_path'.

    Returns:
        dict: A dictionary where keys are 'image_id' and values are the 'file_path' of the
              contralateral image. If no pair is found, the value is None.
    """
    # Create a fast lookup for file paths based on (patient_id, view, laterality)
    # We use a dictionary comprehension for efficiency
    # Key: (patient_id, view, laterality) -> Value: file_path
    path_index = dict(
        zip(zip(df["patient_id"], df["view"], df["laterality"]), df["file_path"])
    )

    contralateral_map = {}

    for _, row in df.iterrows():
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]
        img_id = row["image_id"]

        # Determine the laterality of the required pair
        if lat == "L":
            opp_lat = "R"
        elif lat == "R":
            opp_lat = "L"
        else:
            # Handle cases where laterality might be unknown or invalid
            contralateral_map[img_id] = None
            continue

        # Construct the key to look up the pair
        target_key = (pid, view, opp_lat)

        # Retrieve the path from the index; returns None if not found
        pair_path = path_index.get(target_key)

        contralateral_map[img_id] = pair_path

    return contralateral_map


def find_contralateral_pair(row, df):
    """
    Identifies the file path of the corresponding contralateral image for a single dataframe row.

    This function searches the dataframe for an image with the same 'patient_id' and 'view'
    but opposite 'laterality'.

    Args:
        row (pd.Series or dict): A row containing 'patient_id', 'view', and 'laterality'.
        df (pd.DataFrame): The full dataframe to search for the pair.

    Returns:
        str or None: The file path of the contralateral image if found, otherwise None.
    """
    pid = row["patient_id"]
    view = row["view"]
    lat = row["laterality"]

    # Determine opposite laterality
    if lat == "L":
        opp_lat = "R"
    elif lat == "R":
        opp_lat = "L"
    else:
        return None

    # Filter the dataframe to find the match
    # Note: This operation is O(N) where N is the length of df.
    # For bulk processing, use create_contralateral_lookup instead.
    match = df[
        (df["patient_id"] == pid) & (df["view"] == view) & (df["laterality"] == opp_lat)
    ]

    if not match.empty:
        # Return the file path of the first match found
        return match.iloc[0]["file_path"]
    else:
        return None
