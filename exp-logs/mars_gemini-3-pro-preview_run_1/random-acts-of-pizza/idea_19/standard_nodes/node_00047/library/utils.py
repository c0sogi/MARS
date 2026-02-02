import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SUBMISSION_PATH, RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(request_ids, probabilities, output_path: str = SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        request_ids (list or np.array): List of request_id strings.
        probabilities (list or np.array): List of predicted probabilities (float).
        output_path (str): Full path where the submission CSV will be saved.
                           Defaults to SUBMISSION_PATH from config.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
