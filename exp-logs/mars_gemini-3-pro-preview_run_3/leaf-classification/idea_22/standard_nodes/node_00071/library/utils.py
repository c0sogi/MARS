import os
import random
import numpy as np
import pandas as pd
import torch
import joblib
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_array(data, path):
    """
    Saves a numpy array to a file using .npy format.
    Creates the directory if it does not exist.

    Args:
        data (np.ndarray): The array to save.
        path (str): The file path (should end in .npy).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_array(path):
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The file path.

    Returns:
        np.ndarray or None: The loaded array, or None if the file does not exist.
    """
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def save_model(model, path):
    """
    Saves a model or pipeline object using joblib.

    Args:
        model: The object to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """
    Loads a model or pipeline object using joblib.

    Args:
        path (str): The file path.

    Returns:
        object or None: The loaded model, or None if the file does not exist.
    """
    if os.path.exists(path):
        return joblib.load(path)
    return None


def format_submission(
    test_ids, predictions, class_names, output_path=Config.SUBMISSION_PATH
):
    """
    Formats the predictions into the required submission CSV format.
    Applies probability clipping to avoid log-loss extremes.

    Args:
        test_ids (array-like): List of image IDs corresponding to the predictions.
        predictions (np.ndarray): Matrix of probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of predictions.
        output_path (str): Path to save the final CSV.
    """
    # Clip probabilities to the range [1e-15, 1 - 1e-15] as per metric requirements
    predictions = np.clip(predictions, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)

    # Create DataFrame
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
