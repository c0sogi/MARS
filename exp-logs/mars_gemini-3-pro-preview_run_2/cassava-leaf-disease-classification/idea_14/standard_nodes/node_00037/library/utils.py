import os
import sys
import random
import logging
import shutil
import numpy as np
import torch

from library.config import CFG


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename=os.path.join(CFG.working_dir, "train.log")):
    """
    Creates and configures a logger that outputs to both a file and the console.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(filename, mode="a")
    file_handler.setLevel(logging.INFO)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
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


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint. If is_best is True, copies the file to a 'best_model' version.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best performance so far.
        filepath (str): The path to save the current checkpoint (e.g., 'checkpoint_fold_0.pth').
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    # Save the checkpoint
    torch.save(state, filepath)

    # If it's the best model, create a copy with the appropriate name
    if is_best:
        filename = os.path.basename(filepath)
        # Replace 'checkpoint' with 'best_model' if present, otherwise prepend
        if "checkpoint" in filename:
            best_filename = filename.replace("checkpoint", "best_model")
        else:
            best_filename = "best_model_" + filename

        best_filepath = os.path.join(directory, best_filename)
        shutil.copyfile(filepath, best_filepath)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
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
