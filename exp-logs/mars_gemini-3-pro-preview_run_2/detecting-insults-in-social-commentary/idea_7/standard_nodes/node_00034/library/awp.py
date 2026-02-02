import torch
from library.config import Config
from library.utils import get_logger

logger = get_logger()


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient ascent to flatten the loss landscape.
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
            adv_param (str): The name of the parameters to attack (default: "weight").
            adv_lr (float): The magnitude of the perturbation relative to weight norm.
            adv_eps (float): Epsilon value (often used as a cap or scaling factor).
            start_epoch (int): The epoch to start applying AWP.
            scaler: GradScaler for mixed precision (optional).
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

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        Saves original weights and applies perturbation.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate norms
                grad_norm = torch.norm(param.grad)
                weight_norm = torch.norm(param.data.detach())

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation:
                    # Direction: grad / grad_norm
                    # Magnitude: adv_lr * weight_norm
                    r_at = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )

                    # Save original weights
                    self.backup[name] = param.data.clone()

                    # Apply perturbation
                    param.data.add_(r_at)

    def restore(self):
        """
        Restores the original model weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory
        self.backup = {}
