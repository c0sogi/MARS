import torch
from library.config import CFG


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.

    This class manages the injection of adversarial perturbations into the model weights
    to flatten the loss landscape and improve generalization/robustness.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        adv_param="weight",
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer (unused in this implementation but kept for compatibility).
            adv_param (str): The substring to match parameter names (e.g., 'weight').
            adv_lr (float): The magnitude (learning rate) of the adversarial perturbation.
            adv_eps (float): The epsilon limit for the perturbation relative to weight magnitude.
            start_epoch (int): The epoch to start applying AWP.
            scaler: GradScaler for AMP (unused here as we operate on params directly).
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
        Backs up the current weights and calculates the epsilon bounds for clipping.
        Only backs up parameters that have gradients and match the adv_param filter.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def attack(self):
        """
        Perturbs the model weights based on the current gradients.
        Should be called after loss.backward().
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate perturbation direction
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Scale perturbation by weight magnitude (relative epsilon)
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clip to epsilon ball around original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def restore(self):
        """
        Restores the original weights from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to save memory
        self.backup = {}
        self.backup_eps = {}
