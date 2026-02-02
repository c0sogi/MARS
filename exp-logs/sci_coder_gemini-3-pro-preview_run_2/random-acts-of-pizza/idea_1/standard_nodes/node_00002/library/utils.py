import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import SUBMISSION_PATH


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_submission(request_ids, probabilities, output_path=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (list or np.array): List of request IDs.
        probabilities (list or np.array): List of predicted probabilities (real-valued).
        output_path (str): Path to save the submission file. Defaults to config path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame conforming to submission format
    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
