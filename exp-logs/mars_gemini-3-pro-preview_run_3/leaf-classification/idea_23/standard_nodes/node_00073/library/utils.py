import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across various libraries (random, numpy, torch).
    Configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss after applying the competition-specific
    rescaling and clipping rules.

    The competition metric specifies:
    1. Probabilities are rescaled so each row sums to 1.
    2. Probabilities are clipped to [1e-15, 1-1e-15].

    Args:
        y_true (array-like): Ground truth labels (n_samples,). Can be class indices or names.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # Rescale rows to sum to 1
    # "each row is divided by the row sum"
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential division by zero (though unlikely with proper softmax/probability outputs)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums

    # Clip probabilities
    # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_rescaled, epsilon, 1 - epsilon)

    # Calculate log loss
    # We allow sklearn to infer labels from y_true or assume y_pred covers all classes in order
    return log_loss(y_true, y_pred_clipped)


def save_submission(ids, probs, class_names, output_path=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,Class_1,Class_2,...
    1,0.1,0.9,...

    Args:
        ids (array-like): Image IDs.
        probs (array-like): Predicted probabilities matrix (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns in probs.
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
    """
    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
