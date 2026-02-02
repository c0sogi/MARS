import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(
    ids: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = Config.SUBMISSION_FILE,
) -> None:
    """
    Saves the prediction results to a CSV file in the format required for submission.

    Args:
        ids (np.ndarray or list): List of image IDs.
        probabilities (np.ndarray or list): List of predicted probabilities for 'is_iceberg'.
        output_path (str): The file path where the submission CSV will be saved.
                           Defaults to Config.SUBMISSION_FILE.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create the DataFrame
    # The submission format requires 'id' and 'is_iceberg' columns
    submission_df = pd.DataFrame({"id": ids, "is_iceberg": probabilities})

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
