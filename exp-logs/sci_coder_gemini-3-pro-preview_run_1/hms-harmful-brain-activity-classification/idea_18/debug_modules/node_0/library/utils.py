import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class KLDivLossWithLogits(nn.Module):
    """
    KL Divergence Loss that accepts logits as input and probabilities as target.
    Applies LogSoftmax to logits before computing KL Divergence to ensure
    numerical stability.
    """

    def __init__(self):
        super().__init__()
        # 'batchmean' is the mathematically correct reduction for KL Div
        # when treating the input as a batch of distributions.
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Logits from the model (batch_size, num_classes)
            y_true: Target probabilities (batch_size, num_classes)
        Returns:
            torch.Tensor: The calculated loss.
        """
        # Apply log_softmax to logits to get log-probabilities
        log_prob = F.log_softmax(y_pred, dim=1)
        return self.kl(log_prob, y_true)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss during training and validation.
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
