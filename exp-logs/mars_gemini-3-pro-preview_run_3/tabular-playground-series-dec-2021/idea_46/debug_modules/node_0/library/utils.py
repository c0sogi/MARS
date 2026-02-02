import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, sets CuDNN to deterministic mode (slower).
                              If False, allows CuDNN benchmark (faster).
                              Default is False to align with the strategy of maximizing kernel performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # As per strategy: Disable strict CuDNN determinism to maximize kernel performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used.
    Prioritizes NVIDIA A100 (CUDA) if available.

    Returns:
        torch.device: The device (cuda or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_submission(
    predictions: np.ndarray,
    ids: np.ndarray,
    filepath: str = "./submission/submission.csv",
):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Array of predicted class labels.
        ids (np.ndarray): Array of corresponding Ids.
        filepath (str): The path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Create the DataFrame
    submission = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    # Save to CSV
    submission.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")


def load_data(split: str = "train", base_path: str = "./metadata"):
    """
    Loads the dataset split from the metadata directory.

    Args:
        split (str): The subset to load ('train', 'val', 'test').
        base_path (str): The directory containing the parquet files.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = os.path.join(base_path, f"{split}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_parquet(file_path)
