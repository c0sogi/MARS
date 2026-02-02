import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import SEED, SUBMISSION_DIR


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates).

    Returns:
        float: The computed AUC score.
    """
    try:
        score = roc_auc_score(y_true, y_pred)
        return score
    except ValueError as e:
        print(f"Error computing AUC: {e}")
        return 0.0


def save_submission(request_ids, probabilities, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (list or array-like): The list of request IDs.
        probabilities (list or array-like): The predicted probabilities of receiving pizza.
        filename (str): The name of the output file. Defaults to "submission.csv".
    """
    # Ensure the submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Create the full path
    filepath = os.path.join(SUBMISSION_DIR, filename)

    # Create DataFrame
    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")
