import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_pickle(obj, path):
    """
    Saves a Python object to a pickle file.
    Used for saving models and pipelines.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads a Python object from a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_npy(array, path):
    """
    Saves a numpy array to a .npy file.
    Used for caching deterministic data processing results.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, array)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.
    """
    return np.load(path)


def format_submission(
    ids, probabilities, class_names, output_path=Config.SUBMISSION_PATH
):
    """
    Formats the predictions into a submission CSV file.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (array-like): Matrix of probabilities (shape: [n_samples, n_classes]).
        class_names (list): List of class names corresponding to the columns of probabilities.
        output_path (str): Path to save the CSV file.
    """
    # Ensure inputs are numpy arrays/lists
    probs = np.array(probabilities)
    ids = np.array(ids)

    # Clip probabilities to avoid log loss extremes [1e-15, 1 - 1e-15]
    # As per metric description: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)
    df.insert(0, "id", ids)

    # Ensure output directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
