import os
import random
import numpy as np
import pandas as pd
import torch
from library import config


def seed_everything(seed=config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_submission(test_ids, classes, probabilities, output_path=None):
    """
    Formats the predictions into the required CSV submission format.
    Applies probability clipping to avoid log loss extremes.

    Args:
        test_ids (array-like): List or array of image IDs corresponding to the predictions.
        classes (list): List of class names (species) corresponding to the columns of probabilities.
        probabilities (numpy.ndarray): Matrix of predicted probabilities with shape (n_samples, n_classes).
        output_path (str, optional): Path to save the submission CSV. Defaults to config.SUBMISSION_FILE_PATH.
    """
    if output_path is None:
        output_path = config.SUBMISSION_FILE_PATH

    # Ensure probabilities are a numpy array
    probs = np.array(probabilities)

    # Clip probabilities to avoid extremes of the log function
    # As per task description: max(min(p, 1-10^-15), 10^-15)
    eps = config.PROB_CLIP_EPS
    probs = np.clip(probs, eps, 1 - eps)

    # Create DataFrame
    # The order of columns must match the order of probabilities
    df = pd.DataFrame(probs, columns=classes)

    # Insert 'id' column at the beginning
    # Ensure IDs are integers if they are numeric
    clean_ids = []
    for i in test_ids:
        try:
            clean_ids.append(int(i))
        except (ValueError, TypeError):
            clean_ids.append(i)

    df.insert(0, "id", clean_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
