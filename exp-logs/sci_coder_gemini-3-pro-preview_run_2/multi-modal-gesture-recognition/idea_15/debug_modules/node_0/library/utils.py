import os
import random
import numpy as np
import torch
import nltk
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein distance metric (Error Rate).

    Metric = Sum(Levenshtein(pred, target)) / Sum(len(target))

    Args:
        predictions (list of list of int): Predicted gesture sequences (e.g., [[1, 2], [3]]).
        targets (list of list of int): Ground truth gesture sequences (e.g., [[1, 2], [3]]).

    Returns:
        float: The calculated error rate.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have the same length."
        )

    total_distance = 0
    total_target_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Compute Levenshtein distance between the two sequences of integers
        dist = nltk.edit_distance(pred_seq, target_seq)
        total_distance += dist
        total_target_length += len(target_seq)

    # Avoid division by zero if the validation set has no gestures (unlikely)
    if total_target_length == 0:
        return 0.0

    return total_distance / total_target_length


def save_submission(sample_ids, predictions, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the format: SampleID,Label1,Label2,...

    Args:
        sample_ids (list of str): List of sequence identifiers (e.g., 'Sample00001').
        predictions (list of list of int): List of predicted gesture sequences.
        filename (str): Name of the output file. Defaults to "submission.csv".
    """
    if len(sample_ids) != len(predictions):
        raise ValueError("Sample IDs and predictions must have the same length.")

    output_path = os.path.join(Config.SUBMISSION_DIR, filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for sid, pred_seq in zip(sample_ids, predictions):
            # Convert list of ints to comma-separated string
            # Example format: Session00001,2,12,3
            if pred_seq:
                pred_str = ",".join(map(str, pred_seq))
                line = f"{sid},{pred_str}\n"
            else:
                # Handle case with no predicted gestures
                line = f"{sid}\n"
            f.write(line)

    print(f"Submission saved to {output_path}")
