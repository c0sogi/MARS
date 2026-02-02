import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient ascent to maximize loss,
    improving model robustness and generalization.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer (optional, included for compatibility).
            adv_param (str): Substring to identify parameters to attack (e.g., "weight").
            adv_lr (float): The magnitude of the attack step (learning rate for ascent).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the AWP attack.
        1. Saves current weights.
        2. Computes perturbation based on gradients.
        3. Applies perturbation to weights.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            # Only attack parameters that have gradients and match the filter
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Compute gradient norm and parameter norm
                grad_norm = torch.norm(param.grad)
                data_norm = torch.norm(param.data.detach())

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation: direction * magnitude
                    # Direction: grad / grad_norm
                    # Magnitude: adv_lr * data_norm
                    r_at = self.adv_lr * param.grad / (grad_norm + e) * (data_norm + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Enforce epsilon constraint: |param_new - param_old| <= adv_eps * |param_old|
                    # We clamp the new parameter data to be within the range [old - eps, old + eps]
                    epsilon = self.adv_eps * (self.backup[name].abs() + e)
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - epsilon),
                        self.backup[name] + epsilon,
                    )

    def _save(self):
        """
        Backs up the current model weights.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def restore(self):
        """
        Restores the model weights from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory/reset state for next step
        self.backup = {}
