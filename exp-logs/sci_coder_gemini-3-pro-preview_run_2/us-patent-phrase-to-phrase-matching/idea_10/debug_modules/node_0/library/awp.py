import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to flatten the loss landscape
    and improve generalization performance.
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
            model: The PyTorch model to perturb.
            optimizer: The optimizer (optional, kept for interface compatibility).
            adv_param (str): Substring to filter parameters to perturb (e.g., "weight").
            adv_lr (float): The magnitude of the perturbation step (learning rate for the attack).
            adv_eps (float): The maximum allowed perturbation magnitude (epsilon constraint).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack step.
        1. Identifies parameters to perturb (requires_grad, has grad, matches adv_param).
        2. Backs up the original parameter values.
        3. Computes the perturbation: delta = adv_lr * grad / ||grad||.
        4. Applies the perturbation and projects it within the epsilon ball.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Only perturb parameters that require gradients, have gradients computed,
            # and match the target name (e.g., 'weight').
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Save the original parameter data
                self.backup[name] = param.data.clone()

                grad = param.grad
                grad_norm = torch.norm(grad)

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation direction (Normalized Gradient)
                    # r_at = alpha * grad / ||grad||
                    r_at = self.adv_lr * grad / (grad_norm + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project within epsilon ball (PGD constraint)
                    # Ensures the perturbed weight is within [orig - eps, orig + eps]
                    param.data = torch.max(
                        torch.min(param.data, self.backup[name] + self.adv_eps),
                        self.backup[name] - self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory
        self.backup = {}
        self.backup_eps = {}
