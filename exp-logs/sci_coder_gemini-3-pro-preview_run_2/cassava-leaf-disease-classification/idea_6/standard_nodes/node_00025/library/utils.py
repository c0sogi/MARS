import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import CFG


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    """
    Initializes and returns a logger that outputs to both console and a file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers if they don't exist to avoid duplicate logs
    if not logger.handlers:
        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)

        # File handler
        if filename:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file_handler = logging.FileHandler(filename, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model logits or probabilities.
        target (torch.Tensor): Ground truth labels.
        topk (tuple): Tuple of int, specifying the top-k accuracies to compute.

    Returns:
        list: A list of float values representing the accuracy for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the indices of the top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # Check which predictions match the target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Sum the correct predictions for the top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Calculate accuracy
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
