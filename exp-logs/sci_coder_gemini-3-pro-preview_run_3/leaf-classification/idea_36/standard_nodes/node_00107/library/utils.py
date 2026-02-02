import os
import random
import numpy as np
import torch
import logging
import pickle
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_path):
    """
    Configures logging to output to both a file and the console.

    Args:
        log_path (str): The path to the log file.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Remove existing handlers to avoid duplication
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def save_pickle(data, path):
    """
    Saves a Python object to a pickle file.

    Args:
        data: The object to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        The loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def format_submission(predictions, test_ids, class_names, output_path):
    """
    Formats predictions into a submission CSV file.

    Args:
        predictions (np.ndarray): A (N_samples, N_classes) array of probabilities.
        test_ids (np.ndarray or list): A list or array of test image IDs.
        class_names (list): A list of class names corresponding to the columns of predictions.
        output_path (str): The path where the submission CSV will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the DataFrame
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert the ID column at the beginning
    df.insert(0, "id", test_ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
    logging.info(f"Submission saved to {output_path}")
