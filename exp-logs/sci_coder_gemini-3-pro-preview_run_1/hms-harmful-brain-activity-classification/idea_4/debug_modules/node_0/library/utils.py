import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def kl_divergence_loss(y_pred_logits, y_true):
    """
    Computes the Kullback-Leibler (KL) Divergence loss.

    This function expects raw logits from the model (before Softmax) for numerical
    stability. It applies LogSoftmax internally before computing the KL divergence
    against the provided soft probability targets.

    Args:
        y_pred_logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
        y_true (torch.Tensor): Ground truth soft probabilities of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: The scalar KL divergence loss (averaged over the batch).
    """
    # Ensure targets are on the correct device
    y_true = y_true.to(y_pred_logits.device)

    # Apply LogSoftmax to logits to get log-probabilities
    # shape: (batch_size, num_classes)
    log_probs = F.log_softmax(y_pred_logits, dim=1)

    # Compute KL Divergence
    # reduction='batchmean' is used to average the KL divergence over the batch,
    # which is the mathematically correct way to compute the mean KL divergence.
    loss = F.kl_div(log_probs, y_true, reduction="batchmean")

    return loss
