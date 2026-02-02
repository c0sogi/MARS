import os
import sys
import random
import numpy as np
import torch
import logging


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=False ensures deterministic algorithm selection, though it may be slower
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    """
    Creates a logger that writes to both a file and the console (stdout).
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # File Handler
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

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
        output (torch.Tensor): Model predictions (logits or probabilities).
        target (torch.Tensor): Ground truth labels.
        topk (tuple): Tuple of k values to compute accuracy for (e.g., (1, 5)).

    Returns:
        list: A list of accuracy values (in percentage) for each k in topk.
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
            # Sum up correct predictions for top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
