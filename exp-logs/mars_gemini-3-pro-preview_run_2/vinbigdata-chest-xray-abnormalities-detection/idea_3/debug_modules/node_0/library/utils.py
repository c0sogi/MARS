import sys
import os
import logging
import torch
import numpy as np
import random
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility by calling the library configuration.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def get_logger(log_file):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_fn(batch):
    """
    Custom collate function for object detection DataLoaders.
    Instead of stacking tensors, it returns a tuple of images and a tuple of targets,
    which is the expected format for PyTorch detection models.

    Args:
        batch: List of tuples (image, target)

    Returns:
        tuple: (tuple of images, tuple of targets)
    """
    return tuple(zip(*batch))


def format_prediction_string(boxes, scores, labels):
    """
    Formats bounding box predictions into the submission string format.

    Args:
        boxes (array-like): Bounding boxes in [xmin, ymin, xmax, ymax] format.
        scores (array-like): Confidence scores.
        labels (array-like): Class IDs.

    Returns:
        str: Formatted prediction string. e.g., "11 0.5 100 100 200 200 ..."
             Returns "14 1 0 0 1 1" if no boxes are provided.
    """
    # If no objects detected, return the specific "No finding" string
    if len(boxes) == 0:
        return "14 1 0 0 1 1"

    prediction_strings = []
    for i in range(len(boxes)):
        class_id = int(labels[i])
        score = float(scores[i])
        box = boxes[i]

        # Handle tensor or numpy array
        if hasattr(box, "tolist"):
            box = box.tolist()

        xmin, ymin, xmax, ymax = box

        # Format: class_id confidence xmin ymin xmax ymax
        prediction_strings.append(f"{class_id} {score} {xmin} {ymin} {xmax} {ymax}")

    return " ".join(prediction_strings)
