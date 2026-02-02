import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_multilabel_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Check if y_true and y_pred have consistent shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        score = roc_auc_score(y_true, y_pred, average="macro")

        if np.isnan(score):
            raise ValueError("roc_auc_score returned NaN")
    except ValueError:
        # Fallback for cases where a class might not be present in the batch (e.g., all 0s)
        # Calculate per column and ignore columns that raise errors
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                # Only calculate if there is more than one class present
                if len(np.unique(y_true[:, i])) > 1:
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aucs.append(auc)
            except ValueError:
                pass

        if len(aucs) > 0:
            score = np.mean(aucs)
        else:
            # If no classes could be evaluated, return 0.5 (random guessing)
            score = 0.5

    return score


def save_submission(predictions, test_rec_ids, output_path=Config.PREDICTIONS_PATH):
    """
    Formats and saves the submission file according to the competition requirements.

    The 'Id' column is constructed by multiplying 'rec_id' by 100 and adding the 'species' number.

    Args:
        predictions (np.ndarray): Predicted probabilities of shape (N_samples, N_species).
        test_rec_ids (list or np.ndarray): List of recording IDs corresponding to the samples.
        output_path (str): Path to save the CSV file.
    """
    if len(predictions) != len(test_rec_ids):
        raise ValueError(
            f"Length of predictions ({len(predictions)}) and test_rec_ids ({len(test_rec_ids)}) must match."
        )

    num_species = predictions.shape[1]

    ids = []
    probs = []

    # Iterate through each sample
    for i, rec_id in enumerate(test_rec_ids):
        # Iterate through each species
        for species_idx in range(num_species):
            # Create unique Id: rec_id * 100 + species_number
            row_id = int(rec_id * 100 + species_idx)
            prob = predictions[i, species_idx]

            ids.append(row_id)
            probs.append(prob)

    submission_df = pd.DataFrame({"Id": ids, "Probability": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
