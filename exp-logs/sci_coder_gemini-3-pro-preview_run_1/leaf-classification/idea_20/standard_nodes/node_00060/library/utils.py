import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Seeding PyTorch to ensure deterministic behavior if used
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(ids, probabilities, classes, output_path):
    """
    Saves the predictions to a CSV file in the required format with high precision.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (numpy.ndarray): (N, C) array of predicted probabilities.
        classes (array-like): List of class names corresponding to the columns of probabilities.
        output_path (str): Path to save the CSV file.
    """
    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the DataFrame
    # We assume probabilities are aligned with the order of 'classes'
    submission_df = pd.DataFrame(probabilities, columns=classes)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV
    # float_format='%.16f' is used to preserve double precision (float64)
    # This is critical for minimizing log loss, as standard formatting might
    # truncate values like 0.999999999999999 to 1.0 or 0.999999.
    submission_df.to_csv(output_path, index=False, float_format="%.16f")

    print(f"Submission saved to {output_path}")
    print(f"Submission dimensions: {submission_df.shape}")
