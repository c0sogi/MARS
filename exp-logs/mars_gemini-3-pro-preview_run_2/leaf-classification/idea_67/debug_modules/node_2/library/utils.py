import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, SUBMISSION_PATH, ID_COL


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(preds):
    """
    Clips probabilities to the range [10^-15, 1-10^-15] to avoid extremes in the log function,
    strictly adhering to the competition metric definition.

    Args:
        preds (np.ndarray): Probability matrix.

    Returns:
        np.ndarray: Clipped probability matrix.
    """
    epsilon = 1e-15
    # Formula: max(min(p, 1-10^-15), 10^-15)
    return np.clip(preds, epsilon, 1.0 - epsilon)


def log_loss_metric(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss.

    Args:
        y_true (array-like): True labels (1D array of classes/indices).
        y_pred (array-like): Predicted probabilities (2D array).
        labels (array-like, optional): List of labels to index the matrix.
                                       Useful if y_true doesn't contain all classes found in y_pred.

    Returns:
        float: The log loss score.
    """
    # Sklearn's log_loss (v1.5+) no longer supports the eps argument.
    # The input y_pred is expected to be clipped externally (which we do via clip_probabilities).
    score = log_loss(y_true, y_pred, labels=labels)
    return score


def save_submission(ids, classes, probs, output_path=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (array-like): List of image IDs.
        classes (array-like): List of class names corresponding to the columns of probs.
        probs (np.ndarray): Matrix of predicted probabilities.
        output_path (str): Path to save the submission file. Defaults to config.SUBMISSION_PATH.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame with class names as columns
    submission_df = pd.DataFrame(probs, columns=classes)

    # Insert the ID column at the start
    submission_df.insert(0, ID_COL, ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
