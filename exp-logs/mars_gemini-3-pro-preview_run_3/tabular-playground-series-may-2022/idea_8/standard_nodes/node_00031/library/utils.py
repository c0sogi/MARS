import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SUBMISSION_FILE, ID_COL, TARGET_COL


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, predictions, filename=SUBMISSION_FILE):
    """
    Formats and saves the predictions to a CSV file matching the competition requirements.

    Args:
        ids (array-like): List or array of sample IDs.
        predictions (array-like): List or array of predicted probabilities.
        filename (str): Path to save the submission file. Defaults to config.SUBMISSION_FILE.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame with correct column names
    df = pd.DataFrame({ID_COL: ids, TARGET_COL: predictions})

    # Save to CSV without the index
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
