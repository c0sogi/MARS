import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights to maximize the loss (Gradient Ascent) during training,
    forcing the model to find a flatter, more robust minimum.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
    ):
        """
        Initialize the AWP attacker.

        Args:
            model (torch.nn.Module): The model to apply AWP to.
            optimizer (torch.optim.Optimizer): The optimizer used in training.
            adv_param (str): The substring to identify parameters to perturb (default: "weight").
                             Usually excludes biases and LayerNorm parameters.
            adv_lr (float): The learning rate (magnitude) for the adversarial perturbation.
            adv_eps (float): Small epsilon for numerical stability during normalization.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Internal method to backup the current model parameters.
        Only saves parameters that are eligible for perturbation (have gradients and match adv_param).
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
        Internal method to restore the model parameters from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack.
        1. Saves the current weights using `_save`.
        2. Computes the perturbation based on the gradient direction.
        3. Adds the perturbation to the model weights.
        """
        self._save()
        e = self.adv_eps

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation:
                    # Direction: grad / (norm_grad + e)
                    # Scale: adv_lr * (norm_data + e)
                    # Formula: delta = adv_lr * (grad / |grad|) * |weight|
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Apply perturbation in-place
                    param.data.add_(r_at)
