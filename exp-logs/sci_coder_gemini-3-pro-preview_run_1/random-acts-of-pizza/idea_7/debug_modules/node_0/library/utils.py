import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SUBMISSION_PATH, RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in torch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(request_ids, probabilities, output_path=SUBMISSION_PATH):
    """
    Saves the prediction results to a CSV file in the required format.

    Args:
        request_ids (list or np.array): List of request IDs.
        probabilities (list or np.array): List of predicted probabilities (real-valued).
        output_path (str): Path to save the submission CSV. Defaults to SUBMISSION_PATH from config.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame conforming to the submission format
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
