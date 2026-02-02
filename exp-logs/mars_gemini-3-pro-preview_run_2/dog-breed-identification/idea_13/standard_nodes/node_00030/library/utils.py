import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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


def save_submission(test_ids, probabilities, class_names, output_path: str = None):
    """
    Formats the predictions and saves them to a CSV file in the required submission format.

    Args:
        test_ids (array-like): List or array of test image identifiers.
        probabilities (numpy.ndarray): Matrix of predicted probabilities with shape (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of the probabilities matrix.
        output_path (str): The file path where the submission CSV will be saved. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a dictionary for the DataFrame
    data = {"id": test_ids}

    # Add probability columns
    # We assume class_names are sorted alphabetically as per standard generator behavior,
    # but the caller is responsible for ensuring class_names matches the probability columns.
    for i, class_name in enumerate(class_names):
        data[class_name] = probabilities[:, i]

    # Create DataFrame
    submission_df = pd.DataFrame(data)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
