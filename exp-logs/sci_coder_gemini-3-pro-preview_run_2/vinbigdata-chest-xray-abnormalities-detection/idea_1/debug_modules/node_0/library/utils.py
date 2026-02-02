import os
import sys
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Averager:
    """
    Computes and stores the average and current value.
    Used for tracking loss during training.
    """

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class Logger:
    """
    Simple logger to write logs to a file and print to stdout.
    """

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Clear previous log
        with open(self.log_path, "w") as f:
            f.write("")

    def log(self, message):
        print(message)
        with open(self.log_path, "a+") as f:
            f.write(message + "\n")


def collate_fn(batch):
    """
    Custom collate function for the data loader.

    Args:
        batch: List of tuples (image, target, image_id)

    Returns:
        tuple: (images, targets, image_ids)
        - images: List of image tensors
        - targets: List of target dictionaries
        - image_ids: List of image ID strings
    """
    return tuple(zip(*batch))


def format_submission_string(boxes, scores, labels):
    """
    Formats the predictions into the submission string format.

    Args:
        boxes (np.array or list): Bounding boxes [xmin, ymin, xmax, ymax].
        scores (np.array or list): Confidence scores.
        labels (np.array or list): Class IDs (Model IDs).

    Returns:
        str: Formatted prediction string.
    """
    # If no predictions, return the "No finding" string
    if len(boxes) == 0:
        return "14 1 0 0 1 1"

    prediction_strings = []
    for i in range(len(boxes)):
        # Model classes are 1-14 (0 is background).
        # Dataset classes are 0-13.
        # We need to convert model class back to dataset class for submission.
        # Submission Class ID = Model Class ID - 1
        class_id = int(labels[i]) - 1

        # Ensure class_id is valid (0-13)
        if class_id < 0 or class_id > 13:
            continue

        score = scores[i]
        box = boxes[i]

        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        prediction_strings.append(f"{class_id} {score:.6f} {xmin} {ymin} {xmax} {ymax}")

    if not prediction_strings:
        return "14 1 0 0 1 1"

    return " ".join(prediction_strings)
