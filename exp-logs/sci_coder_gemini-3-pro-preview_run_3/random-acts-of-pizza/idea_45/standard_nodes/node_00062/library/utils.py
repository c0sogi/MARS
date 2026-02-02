import os
import time
import random
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name):
    """
    Context manager to measure and print the execution time of a code block.
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def load_data(split):
    """
    Loads the metadata Parquet file for a specific split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded data.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    file_path = os.path.join(Config.METADATA_DIR, f"{split}.parquet")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    return pd.read_parquet(file_path)


def save_submission(predictions, test_ids, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        predictions (array-like): Predicted probabilities of receiving pizza.
        test_ids (array-like): The corresponding request_ids.
        filename (str): The name of the output file.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    output_path = os.path.join(Config.SUBMISSION_DIR, filename)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
