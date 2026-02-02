import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, predictions, filename=None):
    """
    Saves the submission file in the required format.

    Args:
        ids (list or np.array): List or array of request_ids.
        predictions (list or np.array): List or array of predicted probabilities.
        filename (str): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    if filename is None:
        filename = Config.SUBMISSION_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    submission = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save to CSV without index
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
