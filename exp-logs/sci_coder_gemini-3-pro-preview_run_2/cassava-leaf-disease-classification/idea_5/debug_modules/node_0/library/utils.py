import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
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


def get_logger(log_file):
    """
    Creates a logger that writes to both a file and stdout.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
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


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that handles soft targets (probabilities) created by MixUp/CutMix.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        # x: predictions (logits), target: soft labels (probabilities)
        # Formula: - sum(target * log_softmax(x))
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Handles both hard labels (indices) and soft labels (probabilities).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # If targets are one-hot/soft (MixUp), convert to indices for accuracy calculation
        if target.ndim == 2:
            _, target = target.max(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
