import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations ensure reproducibility but may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name="rsna_mammography"):
    """
    Creates and returns a logger that prints to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def calibrate_probabilities(
    probabilities,
    train_prevalence=Config.TRAIN_PREVALENCE,
    test_prevalence=Config.TEST_PREVALENCE,
):
    """
    Applies analytical log-odds correction to predicted probabilities to account for
    class imbalance shifts between training (balanced) and testing (imbalanced).

    Formula:
        Logit_corrected = Logit_pred + log(P_test / (1 - P_test)) - log(P_train / (1 - P_train))

    Args:
        probabilities (np.ndarray or float): Predicted probabilities (0 to 1).
        train_prevalence (float): Prevalence in the training set.
        test_prevalence (float): Expected prevalence in the test set.

    Returns:
        np.ndarray: Calibrated probabilities.
    """
    # Ensure input is numpy array
    probs = np.asarray(probabilities)

    # Clip probabilities to avoid log(0) or log(1)
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Convert to logits
    logits = np.log(probs / (1 - probs))

    # Calculate correction factor
    logit_test = np.log(test_prevalence / (1 - test_prevalence))
    logit_train = np.log(train_prevalence / (1 - train_prevalence))
    correction = logit_test - logit_train

    # Apply correction
    corrected_logits = logits + correction

    # Convert back to probabilities using sigmoid
    corrected_probs = 1 / (1 + np.exp(-corrected_logits))

    return corrected_probs


def pf1_score(labels, preds):
    """
    Calculates the Probabilistic F1 score (pF1).

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)
        pTP = Sum(preds * labels)
        pFP = Sum(preds * (1 - labels))
        TP + FN = Sum(labels) (Total Positives)
        pTP + pFP = Sum(preds)

    Args:
        labels (np.ndarray): Binary ground truth labels (0 or 1).
        preds (np.ndarray): Predicted probabilities.

    Returns:
        float: The pF1 score.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Probabilistic True Positives
    p_tp = np.sum(labels * preds)

    # Denominator for Precision: Sum of all predicted probabilities
    # pTP + pFP = Sum(y_pred * y_true) + Sum(y_pred * (1 - y_true)) = Sum(y_pred)
    precision_denom = np.sum(preds)

    # Denominator for Recall: Total actual positives
    recall_denom = np.sum(labels)

    # Calculate pPrecision
    if precision_denom == 0:
        p_precision = 0.0
    else:
        p_precision = p_tp / precision_denom

    # Calculate pRecall
    if recall_denom == 0:
        p_recall = 0.0
    else:
        p_recall = p_tp / recall_denom

    # Calculate pF1
    if (p_precision + p_recall) == 0:
        pf1 = 0.0
    else:
        pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)

    return pf1
