import os
import sys
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.seed) -> None:
    """
    Seeds all random number generators to ensure reproducible results.

    Args:
        seed (int): The random seed to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str = None) -> logging.Logger:
    """
    Creates a logger that writes to a file and stdout.

    Args:
        log_file (str, optional): Path to the log file. If None, defaults to
                                  Config.output_dir/train.log.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.output_dir, "train.log")

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter("%(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
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
        output (torch.Tensor): Model predictions (logits).
        target (torch.Tensor): Ground truth labels. Can be indices or soft targets (MixUp).
        topk (tuple): Tuple of k values for top-k accuracy.

    Returns:
        list: List of accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Handle soft targets (e.g., from MixUp/CutMix) by taking argmax
        if target.ndim == 2:
            target = target.argmax(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the output directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        output_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(output_dir, "model_best.pth"))
