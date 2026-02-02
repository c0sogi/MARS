import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights during training to flatten the loss landscape and improve generalization.
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
            model: The PyTorch model to perturb.
            optimizer: The optimizer used for training.
            adv_param (str): The name pattern of parameters to perturb (default: "weight").
            adv_lr (float): The learning rate for the adversarial perturbation.
            adv_eps (float): The maximum magnitude (epsilon) of the perturbation.
            start_epoch (int): The epoch at which to start applying AWP.
            scaler: Optional GradScaler for mixed precision training.
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

    def perturb(self, epoch):
        """
        Applies adversarial perturbation to the model weights based on the current gradients.
        Should be called after the first backward pass and before the second forward pass.

        Args:
            epoch (int): The current training epoch.

        Returns:
            bool: True if perturbation was applied, False otherwise (e.g., if before start_epoch).
        """
        if epoch < self.start_epoch:
            return False

        self._save()
        self._attack_step()
        return True

    def _save(self):
        """
        Backs up the current model weights before perturbation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _attack_step(self):
        """
        Calculates and applies the perturbation.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad.data
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation: lr * (grad / |grad|) * |weight|
                    # This scales the perturbation relative to the weight magnitude (AWP specific)
                    perturbation = (
                        self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Clamp the weights to stay within the epsilon ball of the original weights
                    param.data = torch.max(
                        torch.min(param.data, self.backup_eps[name] + self.adv_eps),
                        self.backup_eps[name] - self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial step and before the optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}
