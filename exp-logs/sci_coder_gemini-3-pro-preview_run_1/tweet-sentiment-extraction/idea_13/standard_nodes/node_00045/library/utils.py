import os
import random
import numpy as np
import torch
import warnings
from library.config import Config

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Ensures that results can be replicated.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    # Disable benchmarking to prevent algorithm selection variance
    torch.backends.cudnn.benchmark = False


def normalize_text(text):
    """
    Normalizes whitespace by collapsing multiple spaces into a single space
    and stripping leading/trailing whitespace.

    This function must be applied to the raw text BEFORE tokenization and
    BEFORE extraction to ensuring the indices align.
    """
    return " ".join(str(text).split())


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.
    Metric: Intersection over Union (IoU) of the set of words.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training/validation loops.
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


def get_score(y_true, y_pred):
    """
    Calculates the average Jaccard score for a list of true and predicted strings.
    """
    return np.mean([jaccard(a, b) for a, b in zip(y_true, y_pred)])
