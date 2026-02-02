import os
import random
import numpy as np
import pandas as pd
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(ids, predictions, output_path=config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (array-like): A list or array of sample IDs.
        predictions (array-like): A list or array of predicted probabilities.
        output_path (str): The file path where the submission CSV will be saved.
                           Defaults to config.SUBMISSION_PATH.
    """
    # Ensure the directory for the output file exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: predictions})

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
