import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KLDivLossWithLogits(nn.Module):
    """
    Kullback-Leibler Divergence Loss wrapper.

    This loss function expects raw model outputs (logits) and probability targets.
    It applies LogSoftmax to the logits before computing the KL Divergence,
    ensuring numerical stability compared to taking the log of softmax probabilities.

    The competition metric is KL Divergence between the predicted probability
    distribution and the observed target probability distribution.
    """

    def __init__(self):
        super(KLDivLossWithLogits, self).__init__()
        # reduction='batchmean' aligns with the mathematical definition of KL divergence
        # averaged over the batch, which is the standard metric implementation.
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL Divergence loss.

        Args:
            logits (torch.Tensor): Raw output from the model of shape (Batch, N_Classes).
            targets (torch.Tensor): Target probabilities of shape (Batch, N_Classes).
                                    Must sum to 1 along the class dimension.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # nn.KLDivLoss expects input to be log-probabilities
        # F.log_softmax is numerically more stable than log(softmax(x))
        log_probs = F.log_softmax(logits, dim=1)

        # Compute loss
        loss = self.kl_loss(log_probs, targets)

        return loss
