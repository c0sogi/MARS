import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) Class.

    This class manages the adversarial training process by:
    1. Saving original weights.
    2. Perturbing weights based on gradient direction to maximize loss.
    3. Restoring original weights after the adversarial backward pass.
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
            adv_param (str): Filter to target specific parameters (e.g., "weight").
            adv_lr (float): The step size (learning rate) for the adversarial attack.
            adv_eps (float): The epsilon constraint (maximum perturbation magnitude).
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
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current model weights before perturbation.
        Only saves parameters that require gradients and match the adv_param filter.
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

    def _restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the weights.

        Logic:
        1. Save original weights.
        2. Calculate perturbation: r = adv_lr * (grad / |grad|) * |weight|
        3. Apply perturbation: weight += r
        4. Clip weight to be within [orig - eps, orig + eps]
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad

                # Calculate norms
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation
                    # We scale the direction (grad/norm_grad) by the weight magnitude (norm_data)
                    # and the adversarial learning rate.
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project the perturbed weight onto the epsilon ball centered at the original weight
                    # This ensures the weight doesn't change too drastically
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )
