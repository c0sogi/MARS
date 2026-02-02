import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SUBMISSION_PATH, SUBMISSION_DIR


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(
    predictions, test_ids, class_names, submission_path=SUBMISSION_PATH
):
    """
    Formats predictions and saves them to a CSV file in the required format.

    Args:
        predictions (numpy.ndarray or torch.Tensor): Matrix of shape (n_samples, n_classes) containing probabilities.
        test_ids (list): List of image IDs corresponding to the predictions.
        class_names (list): List of class names (breeds) corresponding to the columns of predictions.
        submission_path (str, optional): Path to save the CSV. Defaults to the path in config.
    """
    # Ensure predictions are numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Validation
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )
    if len(class_names) != predictions.shape[1]:
        raise ValueError(
            f"Class mismatch: {len(class_names)} classes vs {predictions.shape[1]} prediction columns."
        )

    # Create Dictionary for DataFrame
    data = {"id": test_ids}

    # Add breed columns
    # Assuming class_names corresponds to the column order of predictions
    for i, breed in enumerate(class_names):
        data[breed] = predictions[:, i]

    df = pd.DataFrame(data)

    # Ensure directory exists
    if submission_path:
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        # Save to CSV
        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    return df
