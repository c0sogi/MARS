import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_submission_file(ids, class_names, probs, filename):
    """
    Creates a submission CSV file formatted correctly for the competition.

    Args:
        ids (array-like): List or array of image IDs.
        class_names (list): List of class names (column headers) corresponding to the probability columns.
        probs (array-like): Predicted probabilities matrix with shape (n_samples, n_classes).
        filename (str): Path where the submission CSV will be saved.
    """
    # Ensure probabilities are a numpy array
    probs = np.array(probs)

    # Clip probabilities to the specified range [1e-15, 1-1e-15] to avoid log loss extremes
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)

    # Insert 'id' column at the beginning
    df.insert(0, "id", ids)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to CSV without the index
    df.to_csv(filename, index=False)
    print(f"Submission file saved to {filename}")
