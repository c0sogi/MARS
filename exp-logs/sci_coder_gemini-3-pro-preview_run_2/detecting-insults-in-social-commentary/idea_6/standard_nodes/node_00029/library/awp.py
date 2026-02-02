import torch
from library.config import Config


class AWP:
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
        Adversarial Weight Perturbation (AWP) class.

        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_lr: The learning rate for the adversarial step (step size).
            adv_eps: The epsilon bound for the perturbation (constraint).
            start_epoch: The epoch to start applying AWP.
            scaler: Optional GradScaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Saves the current weights and applies the adversarial perturbation.
        Should be called after the first backward pass (calculating gradients on clean data).
        """
        self._save()
        self._attack_step()

    def restore(self):
        """
        Restores the original weights of the model.
        Should be called after the adversarial backward pass and before the optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the original weights and computes the constraint bounds for parameters that have gradients.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                if name not in self.backup:
                    # Save original weights
                    self.backup[name] = param.data.clone()

                    # Calculate epsilon bounds relative to weight magnitude
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _attack_step(self):
        """
        Computes and applies the perturbation to the weights based on gradients.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Compute perturbation: direction * step_size * weight_scale
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project back onto the epsilon ball (clip to bounds)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
