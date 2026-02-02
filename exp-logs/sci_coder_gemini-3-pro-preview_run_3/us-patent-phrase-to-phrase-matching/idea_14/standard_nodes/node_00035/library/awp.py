import torch
import torch.nn as nn
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("awp.log")


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to maximize loss,
    thereby flattening the loss landscape and improving generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    ):
        """
        Args:
            model (nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): Keyword to filter parameters to perturb (e.g., "weight").
            adv_lr (float): The step size (scaling factor) for the perturbation.
            adv_eps (float): The maximum allowed magnitude of perturbation relative to weight norm.
            start_epoch (int): The epoch to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        Saves the current weights and applies the perturbation.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Filter parameters: must require grad, have grad, and match the adv_param keyword
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Save original weights
                self.backup[name] = param.data.clone()

                # Calculate norms
                grad_norm = torch.norm(param.grad)
                weight_norm = torch.norm(param.data.detach())

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation
                    # Direction: grad / ||grad||
                    # Magnitude: adv_lr * ||weight||
                    r_at = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project back to epsilon ball if necessary
                    # Constraint: ||w_new - w_old|| <= adv_eps * ||w_old||
                    if self.adv_eps > 0:
                        diff = param.data - self.backup[name]
                        diff_norm = torch.norm(diff)
                        max_diff = self.adv_eps * (weight_norm + e)

                        if diff_norm > max_diff:
                            scale = max_diff / (diff_norm + e)
                            param.data = self.backup[name] + diff * scale

    def restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial backward pass and before the optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory
        self.backup = {}
        self.backup_eps = {}

    def should_apply(self, epoch):
        """
        Checks if AWP should be applied based on the current epoch.

        Args:
            epoch (int): The current training epoch (0-indexed).

        Returns:
            bool: True if AWP should be active.
        """
        return epoch >= self.start_epoch
