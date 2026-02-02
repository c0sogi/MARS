import os
import time
import random
import numpy as np
import pandas as pd
import torch
from library.config import SEED


def set_seed(seed: int = SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch settings for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, description: str = "Process"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.description}] Start")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.description}] Done. Duration: {elapsed_time} seconds")


def save_submission(predictions: np.ndarray, test_ids: np.ndarray, output_path: str):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        predictions (np.ndarray): Array of predicted probabilities.
        test_ids (np.ndarray): Array of request IDs corresponding to the predictions.
        output_path (str): File path where the submission CSV should be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Construct the DataFrame
    submission = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV without index
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
