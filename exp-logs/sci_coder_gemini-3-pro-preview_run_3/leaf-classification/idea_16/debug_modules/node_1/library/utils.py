import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log loss extremes,
    as specified in the metric description.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    # The metric requirement: max(min(p, 1-10^-15), 10^-15)
    return np.clip(probs, epsilon, 1.0 - epsilon)


def save_submission(ids, probs, class_names, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the submission CSV file.

    Args:
        ids (array-like): List or array of image IDs.
        probs (np.ndarray): Matrix of probabilities with shape (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of probs.
        output_path (str): File path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure probabilities are clipped according to metric requirements
    probs = clip_probabilities(probs)

    # Create DataFrame
    # Columns must be the class names
    df = pd.DataFrame(probs, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
