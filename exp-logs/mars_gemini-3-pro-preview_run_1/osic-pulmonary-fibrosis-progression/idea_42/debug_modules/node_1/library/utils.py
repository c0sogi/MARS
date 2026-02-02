import os
import random
import numpy as np
import torch
from library.config import MAX_ERROR, MIN_CONFIDENCE


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
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


def laplace_log_likelihood_loss(true_fvc, pred_fvc, pred_sigma):
    """
    Computes the modified Laplace Log Likelihood Loss as defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    This function returns the negative of the metric (Loss) to be minimized:
        Loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (torch.Tensor): Ground truth FVC values.
        pred_fvc (torch.Tensor): Predicted FVC values.
        pred_sigma (torch.Tensor): Predicted confidence (sigma).
                                   Assumed to be positive (e.g. output of Softplus).

    Returns:
        torch.Tensor: The mean loss over the batch.
    """
    # 1. Clip confidence (sigma) at 70 ml
    sigma_clipped = torch.clamp(pred_sigma, min=MIN_CONFIDENCE)

    # 2. Calculate absolute error (delta)
    absolute_error = torch.abs(true_fvc - pred_fvc)

    # 3. Threshold error at 1000 ml
    delta = torch.clamp(absolute_error, max=MAX_ERROR)

    # 4. Compute Loss components
    # We use the device of the input tensor to ensure compatibility
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=true_fvc.device))

    # Term 1: sqrt(2) * delta / sigma
    term_1 = (sqrt_2 * delta) / sigma_clipped

    # Term 2: ln(sqrt(2) * sigma)
    term_2 = torch.log(sqrt_2 * sigma_clipped)

    # 5. Sum terms to get Loss (Negative Metric)
    loss = term_1 + term_2

    # Return mean loss over the batch
    return torch.mean(loss)
