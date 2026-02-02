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


def calculate_consistency(fold_predictions):
    """
    Calculates the mean inter-fold Jaccard consistency for each sample.
    This is used to identify high-confidence samples for pseudo-labeling.

    Args:
        fold_predictions (list of list of str): A list where each element is a list of
                                                predictions for the test set from a specific fold.
                                                Structure: [fold_1_preds, fold_2_preds, ...]
                                                where fold_k_preds is a list of strings.

    Returns:
        np.array: An array of consistency scores for each sample. Shape: (num_samples,)
    """
    num_folds = len(fold_predictions)
    if num_folds < 2:
        # If there's only one fold or no folds, consistency is not applicable in a pairwise sense.
        # Returning 1.0 implies full confidence in the single available prediction.
        if num_folds == 1:
            return np.ones(len(fold_predictions[0]))
        return np.array([])

    # Transpose to iterate sample by sample: zip(*[[s1_f1, s2_f1], [s1_f2, s2_f2]]) -> [(s1_f1, s1_f2), (s2_f1, s2_f2)]
    predictions_by_sample = zip(*fold_predictions)
    consistency_scores = []

    for preds in predictions_by_sample:
        score_sum = 0.0
        pair_count = 0

        # Calculate pairwise Jaccard for all unique pairs of folds
        for i in range(num_folds):
            for j in range(i + 1, num_folds):
                score_sum += jaccard(preds[i], preds[j])
                pair_count += 1

        if pair_count > 0:
            consistency_scores.append(score_sum / pair_count)
        else:
            consistency_scores.append(0.0)

    return np.array(consistency_scores)
