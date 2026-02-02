import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log(0) errors
    in Log Loss calculation, as specified by the task metric.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-16
    # Formula: max(min(p, 1-10^-16), 10^-16)
    return np.clip(probs, epsilon, 1 - epsilon)


def save_submission(ids, class_names, probs, output_path):
    """
    Formats and saves the submission file in the required CSV format.

    Args:
        ids (list or np.ndarray): List of image IDs.
        class_names (list): List of class names corresponding to the columns of probs.
        probs (np.ndarray): Matrix of predicted probabilities (n_samples, n_classes).
        output_path (str): Path to save the CSV file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Validate shapes
    if probs.shape[1] != len(class_names):
        raise ValueError(
            f"Shape mismatch: probs has {probs.shape[1]} columns, "
            f"but class_names has {len(class_names)} elements."
        )

    if len(ids) != probs.shape[0]:
        raise ValueError(
            f"Shape mismatch: ids has {len(ids)} elements, "
            f"but probs has {probs.shape[0]} rows."
        )

    # Construct the DataFrame
    # Start with 'id' column
    data = {"id": ids}

    # Add each class probability as a column
    for i, class_name in enumerate(class_names):
        data[class_name] = probs[:, i]

    df = pd.DataFrame(data)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
