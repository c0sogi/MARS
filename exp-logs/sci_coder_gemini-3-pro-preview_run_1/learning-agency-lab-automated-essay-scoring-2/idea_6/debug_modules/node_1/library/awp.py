import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Implements the logic to perturb model weights in the direction of the gradient
    to maximize loss, thereby flattening the loss landscape and improving generalization.
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
            adv_param (str): The name pattern of parameters to perturb (default: "weight").
            adv_lr (float): The magnitude of the perturbation step (learning rate for the attack).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch to start applying AWP.
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

    def _save(self):
        """
        Save the original weights of the parameters to be perturbed.
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
        Restore the original weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def attack(self):
        """
        Perturb the model weights based on the gradients (Gradient Ascent).
        This method should be called after loss.backward() so that gradients are populated.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Get gradient
                grad = param.grad

                # Calculate norm
                norm = torch.norm(grad)

                # Apply perturbation if norm is valid
                if norm != 0 and not torch.isnan(norm):
                    # Calculate perturbation direction and scale
                    # r_at = alpha * g / ||g||
                    # Note: Scale factor from AMP cancels out in normalization
                    r_at = self.adv_lr * grad / (norm + e)

                    # Add perturbation to weights
                    param.data.add_(r_at)

                    # Project back to epsilon ball around original weights
                    # w' = min(max(w', w_orig - eps), w_orig + eps)
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )
