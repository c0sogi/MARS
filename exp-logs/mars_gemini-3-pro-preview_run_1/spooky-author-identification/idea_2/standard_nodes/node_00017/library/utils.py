import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred, labels=None):
    """
    Computes the multi-class logarithmic loss.

    Args:
        y_true (array-like): Ground truth labels (indices or strings).
        y_pred (array-like): Predicted probabilities.
        labels (list, optional): List of class labels to index the probabilities.

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles label encoding if classes are provided
    return log_loss(y_true, y_pred, labels=labels)


def format_submission(ids, probs, columns):
    """
    Formats the predictions into a DataFrame for submission, applying
    the required probability clipping.

    Args:
        ids (list or array): Sequence of example IDs.
        probs (numpy.ndarray): Predicted probabilities of shape (n_samples, n_classes).
        columns (list): List of column names for the classes (e.g., ['EAP', 'HPL', 'MWS']).

    Returns:
        pd.DataFrame: The formatted submission DataFrame.
    """
    # Apply clipping to avoid extremes of the log function
    # Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

    # Create DataFrame
    submission = pd.DataFrame(probs_clipped, columns=columns)
    submission.insert(0, "id", ids)

    return submission


def generate_meta_features(texts, prob_arrays):
    """
    Generates meta-features for stacking:
    1. Text statistics (length, word count)
    2. Uncertainty metrics from base model probabilities (Entropy, Std, Max)

    Args:
        texts (pd.Series): Input texts.
        prob_arrays (list of np.ndarray): List of probability matrices from base models.

    Returns:
        pd.DataFrame: Meta-features.
    """
    # Cite solution_lesson_node_00008: Meta-Feature Stacking
    # Cite solution_lesson_node_00009: Explicit Uncertainty Signals

    meta = pd.DataFrame()

    # Text Stats
    texts = texts.fillna("").astype(str)
    meta["char_len"] = texts.apply(len)
    meta["word_count"] = texts.apply(lambda x: len(x.split()))

    # Uncertainty Stats for each base model
    for i, probs in enumerate(prob_arrays):
        # Entropy: -sum(p * log(p))
        eps = 1e-15
        p_safe = np.clip(probs, eps, 1 - eps)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)

        # Max probability (Confidence)
        max_prob = np.max(probs, axis=1)

        # Std deviation
        std_prob = np.std(probs, axis=1)

        meta[f"model_{i}_entropy"] = entropy
        meta[f"model_{i}_max_conf"] = max_prob
        meta[f"model_{i}_std"] = std_prob

    return meta
