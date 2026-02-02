import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import ID_COL, TARGET_COL


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)


def verify_gpu():
    """
    Checks and prints the availability of GPU/CUDA.
    Returns True if GPU is available, False otherwise.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Available: True")
        print(f"Device Count: {device_count}")
        print(f"Device Name: {device_name}")
        return True
    else:
        print("GPU Available: False")
        return False


def save_submission(predictions, test_ids, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray or list): Predicted class labels.
        test_ids (np.ndarray or list): Corresponding IDs for the test set.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: predictions})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
