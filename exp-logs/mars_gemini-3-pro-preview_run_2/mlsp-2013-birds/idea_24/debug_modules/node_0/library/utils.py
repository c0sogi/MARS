import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score.

    Handles cases where specific classes might be missing from the ground truth
    batch/subset (e.g., rare bird species not present in the validation set).
    In such cases, those classes are excluded from the average to prevent errors.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Check if both classes (0 and 1) are present in y_true for this label
        # roc_auc_score requires both positive and negative samples
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Fallback for edge cases
                continue

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)


def save_submission(rec_ids, probabilities, output_path):
    """
    Saves predictions to a CSV file in the competition format.

    The submission format requires an 'Id' column constructed as:
    Id = rec_id * 100 + species_id

    Args:
        rec_ids (list or np.array): List of recording IDs.
        probabilities (np.ndarray): Matrix of probabilities (N_samples, N_species).
        output_path (str): Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_rows = []
    num_species = probabilities.shape[1]

    for idx, rec_id in enumerate(rec_ids):
        probs = probabilities[idx]
        for species_id in range(num_species):
            # Construct the unique Id as specified in the task description
            row_id = int(rec_id * 100 + species_id)
            prob = probs[species_id]
            submission_rows.append([row_id, prob])

    df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
    df_sub.to_csv(output_path, index=False)
    # print(f"Submission saved to {output_path}")
