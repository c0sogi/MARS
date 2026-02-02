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
        seed (int): The random seed to apply. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(ids, predictions, output_path: str = None):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (array-like): Array of sample IDs.
        predictions (array-like): Array of predicted probabilities.
        output_path (str, optional): Path to save the CSV file.
                                     If None, uses Config.SUBMISSION_FILE.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_FILE

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
