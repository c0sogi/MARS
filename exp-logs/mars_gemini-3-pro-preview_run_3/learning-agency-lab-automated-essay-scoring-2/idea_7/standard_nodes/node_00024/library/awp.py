import torch
from library.utils import get_logger

# Initialize logger for this module
logger = get_logger("awp.log")


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.

    This class implements the AWP technique which perturbs model weights
    in the direction of the gradient to maximize loss during training.
    This acts as a regularizer, smoothing the loss landscape and improving
    generalization, particularly for large models like DeBERTa.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1e-4,
        adv_eps=1e-2,
        start_epoch=0,
        scaler=None,
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to attack.
            optimizer (optim.Optimizer): The optimizer used in the training loop.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The learning rate (magnitude) for the adversarial perturbation.
            adv_eps (float): The epsilon constraint (maximum allowed perturbation).
            start_epoch (int): The epoch number to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Scaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler

        # Storage for original weights and epsilon constraints
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Internal method to save the current weights of targeted parameters.
        Also calculates the min/max allowed values for the weights based on adv_eps.
        """
        for name, param in self.model.named_parameters():
            # Target only parameters that require gradients, have gradients, and match the pattern
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    # Backup original data
                    self.backup[name] = param.data.clone()

                    # Calculate epsilon range relative to weight magnitude
                    grad_eps = self.adv_eps * param.data.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def attack(self):
        """
        Perturbs the model weights.

        This should be called after the first backward() pass in the training loop.
        It modifies the model weights in-place to maximize the loss.
        """
        e = 1e-6  # Small constant to prevent division by zero
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad

                # Calculate norms
                # Note: If using mixed precision, grad is scaled.
                # However, grad / norm(grad) cancels the scale, preserving the direction.
                norm1 = torch.norm(grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: direction * weight_magnitude * step_size
                    r_at = self.adv_lr * grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation to weights
                    param.data.add_(r_at)

                    # Clamp weights to ensure they stay within epsilon ball of original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def restore(self):
        """
        Restores the original model weights.

        This should be called after the adversarial forward/backward pass and
        before the optimizer step (or after, depending on the specific AWP variant,
        but typically before to ensure the optimizer updates the original weights).
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory for the next step
        self.backup = {}
        self.backup_eps = {}
