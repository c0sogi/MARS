import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_submission_file(predictions, test_ids, output_path):
    """
    Generates the submission CSV file in the required format.

    Args:
        predictions (numpy.ndarray or list): Predicted probabilities for the is_iceberg class.
        test_ids (numpy.ndarray or list): Corresponding IDs for the test images.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure inputs are 1D arrays
    preds = np.array(predictions).flatten()
    ids = np.array(test_ids).flatten()

    # Validate lengths
    if len(preds) != len(ids):
        raise ValueError(
            f"Length mismatch: predictions ({len(preds)}) vs ids ({len(ids)})"
        )

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "is_iceberg": preds})

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def print_metrics(metrics_dict):
    """
    Prints metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
    """
    print("Validation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
