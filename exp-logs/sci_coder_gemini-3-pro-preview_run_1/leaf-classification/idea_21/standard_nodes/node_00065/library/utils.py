import os
import random
import numpy as np
import pandas as pd
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and system environments.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, probabilities, classes, output_path):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        ids (array-like): 1D array or list of image identifiers.
        probabilities (numpy.ndarray): 2D array of predicted probabilities (n_samples, n_classes).
        classes (list): List of class names corresponding to the columns of probabilities.
        output_path (str): Full path where the CSV file will be saved.
    """
    # Ensure the output directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create DataFrame
    # We assume probabilities are already in the correct order matching 'classes'
    df = pd.DataFrame(probabilities, columns=classes)

    # Insert 'id' as the first column
    df.insert(0, "id", ids)

    # Save to CSV
    # Using '%.15f' to preserve float64 precision as per the strategy requirements
    df.to_csv(output_path, index=False, float_format="%.15f")

    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {df.shape}")
