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
        start_epoch=Config.awp_start_epoch,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The magnitude of the attack (step size).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch to start applying AWP.
            scaler: GradScaler for mixed precision training (optional).
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
        Backs up the current model weights.
        Only saves parameters that require gradients, have gradients, and match the adv_param filter.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """
        Restores the backed-up model weights.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the weights.
        1. Saves current weights.
        2. Computes perturbation based on gradient direction and weight magnitude.
        3. Applies perturbation and clips to epsilon ball.
        """
        self._save()
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad

                # Calculate norms
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Compute perturbation: direction * magnitude * scaling
                    # We scale by the weight magnitude (norm_data) to make it relative
                    perturbation = (
                        self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Clip the weights to ensure they stay within epsilon ball of original weights
                    if self.adv_eps > 0:
                        min_value = self.backup[name] - self.adv_eps
                        max_value = self.backup[name] + self.adv_eps
                        # Clamp data to be >= min_value and <= max_value
                        param.data = torch.max(
                            torch.min(param.data, max_value), min_value
                        )

    def restore(self):
        """
        Restores the original weights after the adversarial step.
        """
        self._restore()
