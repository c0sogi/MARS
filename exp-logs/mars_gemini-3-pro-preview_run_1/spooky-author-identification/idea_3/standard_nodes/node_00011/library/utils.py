import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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
    Clips probabilities to avoid logarithmic extremes, as per the metric definition.
    Formula: max(min(p, 1-10^-15), 10^-15)

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1 - epsilon)


def save_submission(ids, probs, output_path=None):
    """
    Formats and saves the submission file.

    Args:
        ids (list or np.ndarray): List of ID strings corresponding to the predictions.
        probs (np.ndarray): Array of shape (n_samples, 3) containing class probabilities.
                            Order of columns must match Config.LABELS.
        output_path (str, optional): Path to save the CSV. If None, uses Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure probs is a numpy array
    probs = np.array(probs)

    # Create DataFrame
    # We assume the probabilities are ordered according to Config.LABELS (EAP, HPL, MWS)
    df = pd.DataFrame(probs, columns=Config.LABELS)

    # Insert the 'id' column at the start
    df.insert(0, "id", ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
