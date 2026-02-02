import torch
from library.config import CFG


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of gradient ascent to flatten the loss landscape,
    improving generalization and robustness.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_param (str): The parameter name pattern to attack (default: "weight").
            adv_lr (float): The magnitude of the perturbation step (scaling factor).
            adv_eps (float): The maximum magnitude of perturbation (epsilon).
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

    def attack(self, epoch):
        """
        Performs the AWP attack:
        1. Saves current weights.
        2. Calculates perturbation based on gradients (Gradient Ascent).
        3. Adds perturbation to weights.

        Args:
            epoch (int): The current training epoch.
        """
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return

        self._save()

        for name, param in self.model.named_parameters():
            # Skip parameters without gradients
            if param.grad is None:
                continue

            # Only attack target parameters (e.g., weights, excluding biases/LN if desired)
            if self.adv_param in name:
                grad = param.grad

                # Calculate norms
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data.detach())

                # Avoid division by zero
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Compute perturbation: eta * (g / ||g||) * ||w||
                    # This scales the perturbation relative to the weight magnitude
                    r_at = self.adv_lr * grad / (norm_grad + 1e-8) * (norm_data + 1e-8)

                    # Apply perturbation to the weights
                    param.data.add_(r_at)

    def _save(self):
        """
        Saves the original values of the parameters that will be attacked.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            if self.adv_param in name:
                # Only save if not already saved (to handle potential nested calls, though unlikely here)
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """
        Restores the original parameter values from the backup and clears the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
