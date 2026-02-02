import logging
import sys
import os
import numpy as np
import torch
from typing import Dict, Union, List, Optional
from library.config import set_seed


def get_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger with the specified name and level.
    Sets up a StreamHandler (stdout) and optionally a FileHandler.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, only logs to stdout.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_class_weights(
    class_counts: Union[Dict[str, int], "pd.Series"],
) -> Dict[str, float]:
    """
    Calculates Square-Root Smoothed Class Weights to handle class imbalance.
    Formula: Weight_c = sqrt(Total_Samples / Count_c)

    This penalizes errors on rare classes more than frequent ones, but less
    aggressively than inverse frequency weighting, preventing gradient instability.

    Args:
        class_counts (dict or pd.Series): Mapping of class label to occurrence count.

    Returns:
        dict: Mapping of class label to calculated weight.
    """
    # Convert pd.Series to dict if necessary
    if hasattr(class_counts, "to_dict"):
        counts = class_counts.to_dict()
    else:
        counts = class_counts

    total_samples = sum(counts.values())
    weights = {}

    for label, count in counts.items():
        if count > 0:
            # Square-root smoothing
            weight = np.sqrt(total_samples / count)
        else:
            # Fallback for classes with 0 count (should not happen in valid training sets)
            weight = 0.0
        weights[label] = weight

    return weights


def weights_to_tensor(
    weights_dict: Dict[str, float], vocab_classes: Dict[str, int], device: str = "cpu"
) -> torch.Tensor:
    """
    Converts a dictionary of class weights into a PyTorch tensor aligned with the class vocabulary indices.
    Useful for passing to CrossEntropyLoss.

    Args:
        weights_dict (dict): Mapping of class label to weight (from calculate_class_weights).
        vocab_classes (dict): Mapping of class label to integer index.
        device (str): Device to place the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A 1D tensor of weights where index i corresponds to the weight for class with index i.
    """
    # Initialize tensor with ones (default weight)
    num_classes = len(vocab_classes)
    weight_tensor = torch.ones(num_classes, dtype=torch.float32)

    for label, idx in vocab_classes.items():
        if label in weights_dict:
            weight_tensor[idx] = weights_dict[label]
        else:
            # If a class in vocab is missing from counts (e.g. never appeared in train),
            # we can leave it as 1.0 or set a high weight.
            # Leaving as 1.0 is safer to avoid exploding gradients for unseen classes.
            pass

    return weight_tensor.to(device)
