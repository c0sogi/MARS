import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to flatten the loss landscape
    and improve generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model to perturb.
            optimizer: The optimizer used for training.
            adv_lr (float): The magnitude of the perturbation step (learning rate for the attack).
            adv_eps (float): The maximum allowed perturbation norm (constraint radius).
            start_epoch (int): The epoch to start applying AWP.
            scaler: Optional GradScaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}

    def _save(self):
        """
        Saves the original weights of the parameters that will be perturbed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """
        Restores the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        1. Saves current weights.
        2. Calculates perturbation based on gradients.
        3. Applies perturbation to weights.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                # Calculate norms
                grad_norm = torch.norm(grad)
                weight_norm = torch.norm(param.data).detach()

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Determine effective step size
                    # We want to step by adv_lr, but not exceed adv_eps constraint relative to weight
                    # Formula: r_at = alpha * (grad / grad_norm) * weight_norm
                    # where alpha is usually adv_lr.
                    # If adv_lr > adv_eps, we clamp to adv_eps.

                    effective_lr = self.adv_lr
                    if self.adv_eps > 0:
                        effective_lr = min(self.adv_lr, self.adv_eps)

                    # Calculate perturbation
                    # We scale by weight_norm to make the perturbation relative to the weight magnitude
                    r_at = effective_lr * grad / (grad_norm + e) * (weight_norm + e)

                    # Apply perturbation
                    param.data.add_(r_at)
