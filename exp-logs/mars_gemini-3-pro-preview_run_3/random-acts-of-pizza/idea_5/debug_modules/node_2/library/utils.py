import os
import time
import random
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from library.config import SEED, SUBMISSION_PATH


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is optional or not installed
        pass


def save_submission(request_ids, predictions, output_path=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (array-like): List of request IDs.
        predictions (array-like): List of predicted probabilities.
        output_path (str): Path to save the CSV.
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def print_metrics(metrics_dict, prefix=""):
    """
    Prints metrics with full precision without rounding.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the print output.
    """
    prefix_str = f"[{prefix}] " if prefix else ""
    for name, value in metrics_dict.items():
        print(f"{prefix_str}{name}: {value}")


@contextmanager
def timer(name):
    """
    Context manager to measure and print execution time of a block.

    Args:
        name (str): Name of the block being measured.
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.2f} s")
