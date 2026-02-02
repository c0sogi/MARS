import torch
import torch.nn as nn
from torch.optim import Optimizer


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.

    This class implements the AWP technique which injects adversarial perturbations
    into the model weights to maximize the loss, thereby regularizing the model
    and finding a flatter minimum in the loss landscape.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1e-4,
        adv_eps: float = 1e-2,
        start_epoch: int = 1,
        scaler=None,
    ):
        """
        Initializes the AWP class.

        Args:
            model (nn.Module): The model to attack.
            optimizer (Optimizer): The optimizer used for training.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The step size for the adversarial perturbation.
            adv_eps (float): The maximum allowed perturbation magnitude (epsilon).
            start_epoch (int): The epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Backs up the current model parameters that will be perturbed.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Save the original parameter value
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    # Pre-compute the allowed range for clipping if needed
                    # (Here we just save the data, clipping logic is in attack_step)
                else:
                    self.backup[name].copy_(param.data)

    def _restore(self):
        """
        Restores the model parameters from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        # Clear backup to save memory if needed, or keep it for efficiency
        self.backup = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the weights.

        1. Saves current weights.
        2. Computes perturbation based on gradients.
        3. Updates weights with perturbation.
        4. Clips weights to be within epsilon ball of original weights.
        """
        e = 1e-6

        # First save the current weights
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Get the gradient
                grad = param.grad

                # Calculate norms
                # Note: If scaler is used, grad is scaled.
                # However, since we normalize grad (grad / grad_norm), the scale cancels out.
                grad_norm = torch.norm(grad)
                weight_norm = torch.norm(param.data)

                # Avoid division by zero
                grad_norm_eff = grad_norm + e

                # Calculate perturbation
                # Direction: grad / grad_norm
                # Magnitude: adv_lr * weight_norm (relative to weight magnitude)
                r_at = self.adv_lr * grad / grad_norm_eff * (weight_norm + e)

                # Apply perturbation
                param.data.add_(r_at)

                # Clip the perturbation to ensure it stays within adv_eps
                # We constrain the new weight to be within [orig - eps*orig, orig + eps*orig]
                # This is a relative constraint commonly used in NLP AWP
                if self.adv_eps > 0:
                    orig = self.backup[name]
                    # Calculate deviation limit
                    limit = self.adv_eps * (weight_norm + e)

                    # Clip
                    # param = max(orig - limit, min(orig + limit, param))
                    # PyTorch equivalent:
                    param.data = torch.min(
                        torch.max(param.data, orig - limit), orig + limit
                    )
