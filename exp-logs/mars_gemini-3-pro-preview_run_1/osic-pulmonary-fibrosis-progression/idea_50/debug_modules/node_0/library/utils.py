import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood Loss as specified in the competition metric.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Since optimizers minimize loss, this module returns -metric:
        loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, pred_fvc, pred_sigma, target_fvc):
        """
        Calculates the loss.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values.
            pred_sigma (torch.Tensor): Predicted Confidence (sigma) values.
            target_fvc (torch.Tensor): True FVC values.

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Ensure inputs are flattened to (N,)
        pred_fvc = pred_fvc.view(-1)
        pred_sigma = pred_sigma.view(-1)
        target_fvc = target_fvc.view(-1)

        # Clip confidence values at 70 ml
        sigma_clipped = torch.clamp(pred_sigma, min=70)

        # Calculate absolute error
        abs_error = torch.abs(target_fvc - pred_fvc)

        # Threshold error at 1000 ml
        delta = torch.clamp(abs_error, max=1000)

        # Calculate loss terms
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=pred_fvc.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Loss is the negative of the metric
        loss = term1 + term2

        return torch.mean(loss)


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
