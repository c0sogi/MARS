import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(
    ids, predictions, output_dir=Config.SUBMISSION_DIR, filename=Config.SUBMISSION_FILE
):
    """
    Saves the submission file in the required format.

    Args:
        ids (array-like): The IDs of the test samples.
        predictions (array-like): The predicted class labels.
        output_dir (str): Directory to save the file. Defaults to Config.SUBMISSION_DIR.
        filename (str): Name of the file. Defaults to Config.SUBMISSION_FILE.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_dir, filename)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    # Save to CSV without the index
    submission_df.to_csv(file_path, index=False)
    print(f"Submission saved to {file_path}")
