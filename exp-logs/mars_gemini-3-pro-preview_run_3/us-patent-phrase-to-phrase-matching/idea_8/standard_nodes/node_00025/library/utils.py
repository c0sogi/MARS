import os
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

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


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # pearsonr returns (statistic, p-value), we only need the statistic
    score = pearsonr(y_true, y_pred)[0]
    return score


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient to maximize loss,
    improving model robustness and generalization.
    """

    def __init__(
        self, model, optimizer, adv_lr=1e-4, adv_eps=1e-4, start_epoch=0, scaler=None
    ):
        """
        Initialize AWP.

        Args:
            model (torch.nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_lr (float): The step size for the adversarial perturbation.
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (int or float): The epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current model weights to a backup dictionary.
        Only saves parameters that have gradients and contain 'weight' in their name.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "weight" in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def attack(self, epoch):
        """
        Performs the adversarial attack (perturbation) on the weights.
        Should be called after the first backward pass.

        Args:
            epoch (int or float): The current training epoch.
        """
        if epoch < self.start_epoch:
            return

        self._save()
        e = 1e-6

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "weight" in name:
                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation:
                    # Direction: grad / norm_grad
                    # Scale: adv_lr * norm_data (relative to weight magnitude)
                    perturbation = (
                        self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Constraint: Keep perturbation within adv_eps of original weight
                    # We use a relative constraint: limit = adv_eps * norm_data
                    limit = self.adv_eps * (norm_data + e)

                    # Clamp the weights
                    param.data = torch.max(
                        torch.min(param.data, self.backup[name] + limit),
                        self.backup[name] - limit,
                    )

    def restore(self):
        """
        Restores the original model weights from backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
