import os
import random
import math
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    where:
        pPrecision = pTP / (pTP + pFP) = Sum(y_true * y_pred) / Sum(y_pred)
        pRecall = pTP / (TP + FN) = Sum(y_true * y_pred) / Sum(y_true)

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities (0 to 1).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Probabilistic True Positives: sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # Probabilistic Precision: pTP / Total Predicted Probability Mass
    # Total Predicted Mass = sum(y_pred)
    p_precision = p_tp / (np.sum(y_pred) + epsilon)

    # Probabilistic Recall: pTP / Total Actual Positives
    # Total Actual Positives = sum(y_true)
    p_recall = p_tp / (np.sum(y_true) + epsilon)

    # Harmonic Mean
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1


def apply_analytical_correction(
    logits,
    train_prevalence=Config.TRAIN_PREVALENCE,
    test_prevalence=Config.TEST_PREVALENCE,
):
    """
    Applies analytical correction to raw logits to account for difference in
    training and test class prevalence.

    Formula:
        L_corrected = L_pred - log(P_train / (1 - P_train)) + log(P_test / (1 - P_test))

    Args:
        logits (torch.Tensor): Raw output logits from the model.
        train_prevalence (float): The proportion of positives in the training set.
        test_prevalence (float): The expected proportion of positives in the test set.

    Returns:
        torch.Tensor: Calibrated probabilities (after sigmoid).
    """
    # Clamp prevalences to avoid log(0) or log(undefined)
    train_p = max(1e-6, min(1 - 1e-6, train_prevalence))
    test_p = max(1e-6, min(1 - 1e-6, test_prevalence))

    # Calculate the log-odds (logit) of the prevalences
    train_logit = math.log(train_p / (1 - train_p))
    test_logit = math.log(test_p / (1 - test_p))

    # Calculate the shift
    shift = -train_logit + test_logit

    # Apply shift to model logits
    # Ensure logits is a tensor if it isn't already
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    corrected_logits = logits + shift

    # Convert to probabilities
    probabilities = torch.sigmoid(corrected_logits)

    return probabilities
