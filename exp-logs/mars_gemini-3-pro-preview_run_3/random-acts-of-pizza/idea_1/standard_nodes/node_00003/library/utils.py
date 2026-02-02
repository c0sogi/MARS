import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to be used for random number generation.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(predictions, test_ids, output_path: str):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        predictions (array-like): Predicted probabilities (or labels) for the test set.
        test_ids (array-like): The corresponding request_ids for the predictions.
        output_path (str): The full file path where the CSV should be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create DataFrame conforming to the submission format
    df = pd.DataFrame({"request_id": test_ids, "requester_received_pizza": predictions})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
