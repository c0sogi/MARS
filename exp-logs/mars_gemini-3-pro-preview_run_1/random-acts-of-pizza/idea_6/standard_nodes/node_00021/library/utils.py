import os
import random
import numpy as np
import torch
import pandas as pd
from library import config


def set_seed(seed=config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_STATE.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(predictions, test_ids, save_path=config.SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        predictions (array-like): Real-valued probabilities of receiving pizza.
        test_ids (array-like): The request_ids corresponding to the predictions.
        save_path (str): The file path to save the CSV. Defaults to config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the submission DataFrame
    # Column names must match the sample submission: request_id, requester_received_pizza
    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV without the index
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
