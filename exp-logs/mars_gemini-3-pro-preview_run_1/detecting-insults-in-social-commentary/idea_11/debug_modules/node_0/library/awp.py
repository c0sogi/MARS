import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to maximize loss,
    smoothing the loss landscape and improving generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
    ):
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The magnitude of the attack step (scaling factor).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Backs up the current parameters of the model that will be attacked.
        Only saves parameters that require gradients and match the adv_param filter.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    # Also initialize backup for epsilon constraint logic if needed
                    # though we calculate it dynamically in attack usually.

    def restore(self):
        """
        Restores the model parameters from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.
        1. Saves current weights.
        2. Calculates perturbation based on gradient direction and weight magnitude.
        3. Applies perturbation and clips within epsilon sphere.
        """
        e = 1e-6  # Small constant for numerical stability
        self._save()

        for name, param in self.model.named_parameters():
            # Filter parameters: must require grad, have grad, and match target name
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                # Skip if gradient is zero or NaN
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation:
                    # Direction: grad / norm_grad
                    # Scale: adv_lr * norm_data (relative to weight magnitude)
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Constraint: Project back to epsilon ball around original weight
                    # param = min(max(param, orig - eps), orig + eps)
                    if self.adv_eps > 0:
                        param.data = torch.min(
                            torch.max(param.data, self.backup[name] - self.adv_eps),
                            self.backup[name] + self.adv_eps,
                        )
