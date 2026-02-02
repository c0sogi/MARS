import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Jaccard similarity score between two strings at the word level.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        float: The Jaccard similarity score (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    # If both sets are empty, define similarity as 0.5 (neutral/undefined overlap)
    # consistent with common evaluation scripts for this task
    if (len(a) == 0) & (len(b) == 0):
        return 0.5

    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_consistency(fold_preds):
    """
    Calculates the consistency (average pairwise Jaccard similarity) of predictions across folds.

    Args:
        fold_preds (list of list of str): A list where each element is a list of predicted strings
                                          for the test set from a specific fold.
                                          Shape: (num_folds, num_samples)

    Returns:
        np.array: An array of consistency scores for each sample. Shape: (num_samples,)
    """
    n_folds = len(fold_preds)
    if n_folds == 0:
        return np.array([])

    n_samples = len(fold_preds[0])
    consistency_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Extract predictions for the i-th sample across all folds
        sample_predictions = [fold_preds[f][i] for f in range(n_folds)]

        # Calculate pairwise Jaccard scores
        scores = []
        for j in range(n_folds):
            for k in range(j + 1, n_folds):
                score = jaccard(sample_predictions[j], sample_predictions[k])
                scores.append(score)

        # Average the pairwise scores
        if scores:
            consistency_scores[i] = np.mean(scores)
        else:
            # If only 1 fold, consistency is 1.0
            consistency_scores[i] = 1.0

    return consistency_scores
