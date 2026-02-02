import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KL_Loss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for classification tasks where targets are probabilities.

    This loss function expects the model predictions to be logits (raw scores).
    It applies LogSoftmax internally before computing the KL Divergence against
    the target probabilities.
    """

    def __init__(self, reduction="batchmean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'batchmean' | 'sum' | 'mean'.
                             'batchmean' is the mathematically correct way to compute
                             KL divergence over a batch. Default: 'batchmean'.
        """
        super(KL_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Computes the KL Divergence loss.

        Args:
            y_pred (torch.Tensor): Predicted logits of shape (Batch, Num_Classes).
            y_true (torch.Tensor): Target probabilities of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Apply LogSoftmax to logits to get log-probabilities
        log_probs = F.log_softmax(y_pred, dim=1)

        # Compute KL Divergence
        # Note: nn.KLDivLoss expects input to be log-probabilities and target to be probabilities
        loss = F.kl_div(log_probs, y_true, reduction=self.reduction)

        return loss
