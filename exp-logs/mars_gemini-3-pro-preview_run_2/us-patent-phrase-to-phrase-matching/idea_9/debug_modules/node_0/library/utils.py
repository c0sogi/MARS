import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient.

    Args:
        y_true: Array-like of ground truth scores.
        y_pred: Array-like of predicted scores.

    Returns:
        float: Pearson correlation coefficient.
    """
    score = pearsonr(y_true, y_pred)[0]
    return score


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


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient to maximize loss,
    improving model robustness and generalization.
    """

    def __init__(
        self, model, optimizer, adv_lr=1e-4, adv_eps=1e-2, start_epoch=0, scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model parameters.
        Saves original weights and applies perturbation.
        """
        self._save()
        for name, param in self.model.named_parameters():
            # Apply perturbation only to parameters that require gradients and have gradients
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                # If using AMP scaler, gradients might be scaled.
                # Normalization (grad / norm) cancels the scale factor for direction,
                # so we can use the scaled gradient directly for direction.
                norm = torch.norm(grad)

                if norm > 1e-6:
                    # Calculate perturbation: direction * step_size
                    r_at = self.adv_lr * grad / (norm + 1e-6)
                    param.data.add_(r_at)

                    # Project perturbation to be within epsilon ball of original weights
                    if self.adv_eps > 0:
                        param.data = torch.max(
                            torch.min(param.data, self.backup[name] + self.adv_eps),
                            self.backup[name] - self.adv_eps,
                        )

    def _save(self):
        """
        Saves the current model weights before perturbation.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """
        Restores the original model weights after the adversarial step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
