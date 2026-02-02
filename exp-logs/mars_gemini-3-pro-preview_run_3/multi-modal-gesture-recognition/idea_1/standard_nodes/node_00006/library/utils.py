import os
import sys
import random
import logging
import numpy as np
import torch
import nltk
from library.config import Config


def set_seeds(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name="experiment", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream Handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_levenshtein_distance(ground_truth, predictions):
    """
    Computes the Levenshtein distance metric as defined in the competition.

    Metric = Sum(Levenshtein(truth_i, pred_i)) / Sum(len(truth_i))

    Args:
        ground_truth (list of list of int): True gesture sequences.
        predictions (list of list of int): Predicted gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_truth_length = 0

    # Ensure lists are of same length
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Length mismatch: GT={len(ground_truth)}, Preds={len(predictions)}"
        )

    for true_seq, pred_seq in zip(ground_truth, predictions):
        # Calculate edit distance
        # nltk.edit_distance works efficiently with lists of integers
        dist = nltk.edit_distance(true_seq, pred_seq)
        total_distance += dist
        total_truth_length += len(true_seq)

    if total_truth_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_truth_length


def pad_collate(batch):
    """
    Collate function for PyTorch DataLoader to handle variable-length sequences.
    Pads features with 0 and labels with -100 (standard ignore index).

    Args:
        batch (list): List of tuples (features, labels).
                      features: Tensor of shape (SeqLen, InputDim)
                      labels: Tensor of shape (SeqLen,)

    Returns:
        tuple: (features_padded, labels_padded, lengths)
            features_padded: (Batch, MaxLen, InputDim)
            labels_padded: (Batch, MaxLen)
            lengths: (Batch,)
    """
    # Separate features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Get lengths
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)

    # Pad features
    # features are tensors of shape (L, D), pad with 0
    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )

    # Pad labels
    # labels are tensors of shape (L,), pad with -100 (common ignore index for CrossEntropyLoss)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return features_padded, labels_padded, lengths
