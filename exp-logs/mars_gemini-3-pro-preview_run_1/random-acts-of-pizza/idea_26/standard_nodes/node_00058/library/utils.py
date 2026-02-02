import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def save_submission(request_ids, predictions, file_path: str):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        request_ids (list or np.array): List of request identifiers.
        predictions (list or np.array): List of predicted probabilities.
        file_path (str): The output path for the CSV file.
    """
    if len(request_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: request_ids ({len(request_ids)}) vs predictions ({len(predictions)})"
        )

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV
    submission_df.to_csv(file_path, index=False)
    print(f"Submission saved to {file_path}")
