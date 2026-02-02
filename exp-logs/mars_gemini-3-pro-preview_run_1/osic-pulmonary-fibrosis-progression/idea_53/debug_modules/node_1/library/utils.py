import os
import random
import numpy as np
import torch
import torch.nn as nn
import math


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss for optimization.

    The competition metric is:
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    To maximize this metric, we minimize the negative:
        Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

    where:
        sigma_clipped = max(sigma, 70)
        delta = |true - pred|  (Note: The metric clips delta at 1000, but for training
                                gradients we typically use unclipped or soft-clipped error)
    """

    def __init__(
        self, clip_sigma=70.0, clip_error=1000.0, apply_error_clip_in_train=False
    ):
        super().__init__()
        self.clip_sigma = clip_sigma
        self.clip_error = clip_error
        self.apply_error_clip_in_train = apply_error_clip_in_train
        # Register constants as buffers or use python scalars in forward to avoid device mismatch
        self.sqrt_2 = math.sqrt(2.0)

    def forward(self, pred_fvc, pred_sigma, target_fvc):
        """
        Args:
            pred_fvc (Tensor): Predicted FVC values.
            pred_sigma (Tensor): Predicted confidence (sigma) values.
            target_fvc (Tensor): Ground truth FVC values.

        Returns:
            Tensor: Scalar loss value (mean over batch).
        """
        # Ensure sigma is at least the clip value (approximate measurement uncertainty)
        # We use clamp to maintain gradients for values > clip_sigma
        sigma_clipped = torch.clamp(pred_sigma, min=self.clip_sigma)

        # Calculate absolute error
        delta = torch.abs(target_fvc - pred_fvc)

        # Optionally clip error during training.
        # Usually False to allow the model to learn from large errors.
        if self.apply_error_clip_in_train:
            delta = torch.clamp(delta, max=self.clip_error)

        # Calculate Loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Loss is the sum
        loss = term1 + term2

        return torch.mean(loss)


def calculate_metric(pred_fvc, pred_sigma, target_fvc):
    """
    Calculates the exact competition metric for evaluation purposes.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        score = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        pred_fvc (Tensor or array): Predicted FVC.
        pred_sigma (Tensor or array): Predicted Confidence.
        target_fvc (Tensor or array): True FVC.

    Returns:
        float: The mean metric score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(pred_fvc, np.ndarray):
        pred_fvc = torch.from_numpy(pred_fvc)
    if isinstance(pred_sigma, np.ndarray):
        pred_sigma = torch.from_numpy(pred_sigma)
    if isinstance(target_fvc, np.ndarray):
        target_fvc = torch.from_numpy(target_fvc)

    # Constants
    clip_sigma = 70.0
    clip_error = 1000.0
    sqrt_2 = math.sqrt(2.0)

    # Apply metric-specific clipping
    sigma_clipped = torch.clamp(pred_sigma, min=clip_sigma)
    delta = torch.abs(target_fvc - pred_fvc)
    delta = torch.clamp(delta, max=clip_error)

    # Calculate metric
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric).item()


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=7, mode="max", delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when
                the quantity monitored has stopped increasing.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta

        if self.mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        score = epoch_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
        else:
            if self.mode == "min":
                improved = score < (self.best_score - self.delta)
            else:
                improved = score > (self.best_score + self.delta)

            if improved:
                self.best_score = score
                self.save_checkpoint(score, model, model_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model, model_path):
        """Saves model when validation score decreases."""
        torch.save(model.state_dict(), model_path)
        self.val_score = score
