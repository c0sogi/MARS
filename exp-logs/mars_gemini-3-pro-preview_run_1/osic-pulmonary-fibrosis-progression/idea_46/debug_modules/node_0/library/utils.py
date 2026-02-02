import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_loss(y_true, y_pred_fvc, y_pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood loss for training.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, we minimize the negative metric (Loss):
        loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): Ground truth FVC values.
        y_pred_fvc (torch.Tensor): Predicted FVC values.
        y_pred_sigma (torch.Tensor): Predicted confidence (sigma) values.

    Returns:
        torch.Tensor: The mean loss over the batch.
    """
    # Ensure inputs are flattened to 1D vectors to prevent broadcasting errors
    y_true = y_true.view(-1)
    y_pred_fvc = y_pred_fvc.view(-1)
    y_pred_sigma = y_pred_sigma.view(-1)

    # Ensure device consistency
    device = y_true.device

    # Constants from Config
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
    max_err = torch.tensor(Config.METRIC_ERROR_CLIP, device=device)
    min_conf = torch.tensor(Config.METRIC_CONFIDENCE_CLIP, device=device)

    # Calculate Delta (Absolute Error)
    delta = torch.abs(y_true - y_pred_fvc)

    # Clip Delta at 1000 ml as per metric definition
    delta_clipped = torch.clamp(delta, max=max_err)

    # Clip Confidence at 70 ml as per metric definition
    # Note: y_pred_sigma is assumed to be positive (e.g., via Softplus activation in the model)
    sigma_clipped = torch.clamp(y_pred_sigma, min=min_conf)

    # Calculate Loss components
    # Term 1: (sqrt(2) * delta) / sigma
    term_1 = (sqrt_2 * delta_clipped) / sigma_clipped

    # Term 2: ln(sqrt(2) * sigma)
    term_2 = torch.log(sqrt_2 * sigma_clipped)

    # Total Loss = Term 1 + Term 2
    loss = term_1 + term_2

    return torch.mean(loss)
