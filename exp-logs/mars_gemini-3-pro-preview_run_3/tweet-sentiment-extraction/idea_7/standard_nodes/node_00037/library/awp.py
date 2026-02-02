import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights to maximize loss during training, which helps the model
    find a flatter minimum in the loss landscape and improves generalization.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    ):
        """
        Initialize the AWP class.

        Args:
            model (torch.nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to None.
            adv_param (str): The parameter name substring to target for perturbation.
                             Defaults to "weight" (targets most layers).
            adv_lr (float): The step size (learning rate) for the adversarial perturbation.
            adv_eps (float): The maximum magnitude (epsilon) of the perturbation.
            start_epoch (int): The epoch at which to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Backs up the current model weights.
        Only saves parameters that:
        1. Require gradients.
        2. Have computed gradients (from the previous backward pass).
        3. Match the `adv_param` filter.
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
        Restores the model weights from the backup.
        Clears the backup to save memory after restoration.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        1. Saves the current (clean) weights.
        2. Calculates the perturbation based on the gradient direction.
        3. Applies the perturbation to the weights.
        4. Projects the perturbed weights to ensure they stay within the epsilon ball
           centered at the original weights.
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
                norm = torch.norm(grad)
                if norm != 0 and not torch.isnan(norm):
                    # Calculate perturbation: direction * step_size
                    r_at = self.adv_lr * grad / (norm + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project back to epsilon ball centered at original weights
                    # min(max(w_new, w_orig - eps), w_orig + eps)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Public method to restore the clean weights.
        Should be called after the adversarial forward/backward pass.
        """
        self._restore()

    def should_apply(self, epoch):
        """
        Checks if AWP should be applied at the given epoch.

        Args:
            epoch (int): The current training epoch (0-indexed).

        Returns:
            bool: True if AWP should be active, False otherwise.
        """
        return epoch >= self.start_epoch
