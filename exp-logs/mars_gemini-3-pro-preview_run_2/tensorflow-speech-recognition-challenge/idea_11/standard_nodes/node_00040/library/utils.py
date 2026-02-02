import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(predictions, test_df, save_path):
    """
    Formats predictions and saves them to a CSV file for submission.

    Args:
        predictions (list or np.ndarray): A list or array of predicted class labels (strings).
        test_df (pd.DataFrame): The test metadata DataFrame containing the 'fname' column.
        save_path (str): The file path where the submission CSV should be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Create the submission DataFrame
    # The competition requires columns: 'fname', 'label'
    submission = pd.DataFrame({"fname": test_df["fname"], "label": predictions})

    # Save to CSV without the index
    submission.to_csv(save_path, index=False)
