import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Wraps the model and optimizer to inject adversarial perturbations into weights
    during training, smoothing the loss landscape and improving robustness.
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
            adv_param (str): Substring to identify parameters to attack (default: "weight").
            adv_lr (float): Magnitude of the attack step (learning rate for ascent).
            adv_eps (float): Maximum allowed perturbation (epsilon).
            start_epoch (int): Epoch to start applying AWP.
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
        Saves the current model weights and calculates the clipping bounds
        for the perturbation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    # Calculate epsilon relative to the weight magnitude
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        """
        Restores the original model weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        """
        Performs the adversarial attack on the weights.
        Perturbation is scaled relative to the gradient and weight norms.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Compute perturbation: direction * step_size * weight_scale
                    # We normalize gradient by its norm and scale by weight norm
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clip weights to ensure they stay within epsilon neighborhood of original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def attack(self):
        """
        Public method to trigger the attack. Saves weights and applies perturbation.
        Should be called after the first backward pass.
        """
        self._save()
        self._attack_step()

    def restore(self):
        """
        Public method to restore weights.
        Should be called after the second backward pass (on perturbed weights).
        """
        self._restore()
