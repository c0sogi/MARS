import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.

    This class handles the logic for:
    1. Saving original model weights.
    2. Perturbing weights in the direction of the gradient to maximize loss.
    3. Restoring original weights after the adversarial step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        optimizer: torch.optim.Optimizer = None,
        adv_param: str = "weight",
    ):
        """
        Initialize the AWP handler.

        Args:
            model (torch.nn.Module): The PyTorch model to attack.
            config (Config): Configuration object containing AWP hyperparameters.
            optimizer (torch.optim.Optimizer, optional): The optimizer.
            adv_param (str): The parameter name substring to target (default: "weight").
                             This usually excludes biases and LayerNorm parameters.
        """
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = config.awp_lr
        self.adv_eps = config.awp_eps

        # Storage for original weights
        self.backup = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.

        1. Saves the current (original) weights.
        2. Calculates the perturbation based on gradients.
        3. Applies the perturbation to the model weights.
        """
        self._save()
        self._attack_step()

    def restore(self):
        """
        Restores the original weights of the model from the backup.
        Should be called after the adversarial backward pass and before the optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory
        self.backup = {}

    def _save(self):
        """
        Saves the current weights of the model to the backup dictionary.
        Only saves parameters that require gradients, have gradients, and match adv_param.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _attack_step(self):
        """
        Calculates and applies the perturbation to the weights.

        Perturbation logic:
        delta = adv_lr * grad / (norm(grad) + epsilon)
        weight_new = weight_old + delta
        weight_new = clamp(weight_new, weight_old - adv_eps, weight_old + adv_eps)
        """
        e = 1e-6  # Small constant to prevent division by zero

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

                    # Project onto the epsilon ball (clip perturbation)
                    # Ensures the weight doesn't drift too far from the original value
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )
