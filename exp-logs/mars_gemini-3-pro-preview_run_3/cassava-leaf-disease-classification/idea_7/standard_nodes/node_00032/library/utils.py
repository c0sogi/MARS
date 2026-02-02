import os
import sys
import logging
import torch
import shutil
from library.config import CFG, seed_everything


def get_logger(filename):
    """
    Creates and configures a logger that writes to both a file and stdout.

    Args:
        filename (str): The path to the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def save_checkpoint(state, is_best, filepath, best_filepath="best_model.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the current checkpoint.
        best_filepath (str): Filename to save the best model (relative to filepath directory).
    """
    torch.save(state, filepath)
    if is_best:
        dirname = os.path.dirname(filepath)
        best_path = os.path.join(dirname, best_filepath)
        shutil.copyfile(filepath, best_path)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
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
        topk (tuple): Tuple of k values for top-k accuracy.

    Returns:
        list: List of computed accuracies for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
