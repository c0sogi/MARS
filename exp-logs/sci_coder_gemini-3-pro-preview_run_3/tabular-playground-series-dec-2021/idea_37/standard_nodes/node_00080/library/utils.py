import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED, CUDNN_DETERMINISTIC, SUBMISSION_FILE


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CuDNN determinism based on the global configuration.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure CuDNN based on the config setting
    if CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # As per Lesson 00070, disabling strict determinism can improve kernel performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_submission(predictions, test_ids, save_path=SUBMISSION_FILE):
    """
    Formats the predictions and test IDs into a DataFrame and saves it as a CSV.

    Args:
        predictions (array-like): The predicted class labels.
        test_ids (array-like): The corresponding IDs for the test set.
        save_path (str): The file path to save the submission CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create DataFrame conforming to the submission format
    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save to CSV without the index
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
