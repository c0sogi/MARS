import os
import random
import numpy as np
import torch
import pandas as pd
import joblib
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to [1e-15, 1-1e-15] to avoid log loss extremes.

    Args:
        probs (np.ndarray or torch.Tensor): The probability matrix.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # Metric requirement: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1.0 - epsilon)


def save_numpy(data, filename):
    """
    Saves a numpy array to the working directory.

    Args:
        data (np.ndarray): Data to save.
        filename (str): Filename (e.g., 'features.npy').
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, data)


def load_numpy(filename):
    """
    Loads a numpy array from the working directory.

    Args:
        filename (str): Filename to load.

    Returns:
        np.ndarray or None: The loaded data or None if file doesn't exist.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(filepath):
        return np.load(filepath, allow_pickle=True)
    return None


def save_parquet(df, filename):
    """
    Saves a pandas DataFrame to parquet in the working directory.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Filename (e.g., 'data.parquet').
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet(filename):
    """
    Loads a pandas DataFrame from parquet in the working directory.

    Args:
        filename (str): Filename to load.

    Returns:
        pd.DataFrame or None: The loaded DataFrame or None if file doesn't exist.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def save_model(model, filename, model_type="sklearn"):
    """
    Saves a model (sklearn or torch) to the working directory.

    Args:
        model: The model object.
        filename (str): Filename for the model.
        model_type (str): 'sklearn' or 'torch'.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if model_type == "torch":
        torch.save(model.state_dict(), filepath)
    elif model_type == "sklearn":
        joblib.dump(model, filepath)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_model(model, filename, model_type="sklearn"):
    """
    Loads a model from the working directory.

    Args:
        model: The instantiated model architecture (required for torch).
        filename (str): Filename to load.
        model_type (str): 'sklearn' or 'torch'.

    Returns:
        The loaded model or None if file doesn't exist.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        return None

    if model_type == "torch":
        model.load_state_dict(torch.load(filepath, map_location=Config.DEVICE))
        return model
    elif model_type == "sklearn":
        return joblib.load(filepath)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def save_submission(ids, probs, filename="submission.csv"):
    """
    Saves the submission file in the correct format.

    Args:
        ids (list or np.array): List of IDs.
        probs (np.array): Probability matrix (N, 3) for EAP, HPL, MWS.
        filename (str): Output filename.
    """
    # Ensure probs are clipped
    probs = clip_probabilities(probs)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
    df.insert(0, "id", ids)

    # Save
    filepath = os.path.join(Config.SUBMISSION_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
