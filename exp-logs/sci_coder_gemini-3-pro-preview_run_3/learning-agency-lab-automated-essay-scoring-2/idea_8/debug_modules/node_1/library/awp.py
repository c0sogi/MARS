import torch
import torch.nn as nn
from library.utils import get_logger

logger = get_logger()


class AWP:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1e-4,
        adv_eps: float = 1e-2,
    ):
        """
        Adversarial Weight Perturbation (AWP) implementation.

        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_param: The parameter name pattern to attack (default: "weight").
            adv_lr: The magnitude of the attack step (learning rate).
            adv_eps: The maximum allowed perturbation (epsilon).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

        logger.info(f"AWP Initialized: lr={adv_lr}, eps={adv_eps}, param={adv_param}")

    def attack_step(self):
        """
        Performs the AWP attack:
        1. Saves current weights.
        2. Computes adversarial perturbation based on gradients.
        3. Updates model weights with perturbation.
        4. Clips weights to stay within epsilon neighborhood of original weights.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate Gradient Norm
                grad_norm = torch.norm(param.grad)

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Direction: grad / grad_norm
                    # Step: adv_lr
                    # Perturbation: adv_lr * (grad / grad_norm)
                    perturbation = self.adv_lr * param.grad / (grad_norm + e)

                    # Apply perturbation (Ascent)
                    param.data.add_(perturbation)

                    # Clip parameters
                    # We ensure the new weight is within [orig - eps, orig + eps]
                    # Using the pre-calculated bounds from _save
                    lower_bound = self.backup_eps[name][0]
                    upper_bound = self.backup_eps[name][1]

                    param.data = torch.max(
                        torch.min(param.data, upper_bound), lower_bound
                    )

    def _save(self):
        """
        Internal method to backup current weights and calculate clipping bounds.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    # Save original weight
                    self.backup[name] = param.data.clone()

                    # Calculate epsilon bounds
                    # Using relative epsilon: eps * |weight|
                    # This scales the constraint with the magnitude of the weight,
                    # which is generally more robust for different layers (embeddings vs attention).
                    grad_eps = self.adv_eps * param.abs().detach()

                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        """
        Restores the original weights from backup and clears the backup storage.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}
