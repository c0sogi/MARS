import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights based on the gradient direction to flatten the loss landscape
    and improve model generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_param (str): The substring to identify parameters to perturb (default: "weight").
            adv_lr (float): The magnitude of the perturbation step (learning rate for AWP).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        Calculates perturbation based on gradients and applies it to the weights.
        Saves original weights to allow restoration.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Apply perturbation only to parameters that:
            # 1. Require gradients
            # 2. Have computed gradients
            # 3. Match the target parameter name (e.g., "weight" to avoid bias/layernorm if desired, though "weight" is standard)
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Compute perturbation:
                    # Direction: param.grad / norm1
                    # Scale: adv_lr * norm2 (weight-relative perturbation)
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Backup original data
                    self.backup[name] = param.data.clone()

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clip the perturbation to ensure it doesn't exceed adv_eps (Projected Gradient Descent style)
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the backward pass on the perturbed loss.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}
        self.backup_eps = {}
