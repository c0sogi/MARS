import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, PROB_CLIP_EPS, FLOAT_PRECISION, SUBMISSION_DIR


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Attempt to set torch seed if available, but do not fail if not installed
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def clip_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss after applying the specific normalization
    and clipping rules defined in the task description.

    The rules are:
    1. Rescale probabilities so each row sums to 1.
    2. Clip probabilities to [1e-15, 1 - 1e-15].
    3. Calculate log loss.

    Args:
        y_true (array-like): True class labels (1D array of shape (n_samples,)).
        y_pred (array-like): Predicted probabilities (2D array of shape (n_samples, n_classes)).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays with correct precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Rescale prior to being scored (each row is divided by the row sum)
    # Handle potential division by zero if a row sums to 0 (though unlikely with valid models)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero by replacing 0 sums with 1 (result remains 0, which is handled by clip)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Replace predicted probabilities with max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = np.clip(y_pred_norm, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    # 3. Calculate Multi-class log loss
    # sklearn log_loss handles the internal log calculations
    score = log_loss(y_true, y_pred_clipped)

    return score


def save_submission(ids, classes, probs, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,Class1,Class2,...
    2,0.1,0.5,...

    Args:
        ids (array-like): List or array of image IDs.
        classes (array-like): List of class names (column headers).
        probs (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        filename (str): Name of the output file. Defaults to "submission.csv".
    """
    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Construct DataFrame
    df = pd.DataFrame(probs, columns=classes)
    df.insert(0, "id", ids)

    # Construct full path
    output_path = os.path.join(SUBMISSION_DIR, filename)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
