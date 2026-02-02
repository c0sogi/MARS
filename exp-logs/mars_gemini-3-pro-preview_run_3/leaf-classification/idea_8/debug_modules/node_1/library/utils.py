import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid log loss extremes
    as specified in the metric description.

    Args:
        probs (np.ndarray or pd.DataFrame): The probability matrix.

    Returns:
        np.ndarray or pd.DataFrame: The clipped probabilities.
    """
    epsilon = 1e-15
    # np.clip works for both numpy arrays and pandas dataframes
    return np.clip(probs, epsilon, 1.0 - epsilon)


def save_submission(ids, probabilities, class_names, filename=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.ndarray): List of image IDs.
        probabilities (np.ndarray): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the probability columns.
        filename (str): Path to save the submission file. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure probabilities are clipped to valid range
    clipped_probs = clip_probabilities(probabilities)

    # Create DataFrame with class columns
    submission_df = pd.DataFrame(clipped_probs, columns=class_names)

    # Insert 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to CSV without the index
    submission_df.to_csv(filename, index=False)
