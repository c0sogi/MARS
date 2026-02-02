import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) module.

    This class manages the adversarial attack process by perturbing model weights
    in the direction of the gradient to maximize loss (ascent), thereby regularizing
    the model and improving robustness.
    """

    def __init__(self, model, optimizer, config: Config):
        """
        Initialize the AWP module.

        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training (stored for reference).
            config: Configuration object containing AWP hyperparameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = config.awp_lr
        self.adv_eps = config.awp_eps
        self.start_epoch = config.awp_start_epoch

        # storage for original weights
        self.backup = {}
        # storage for reference weights for epsilon constraint
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.
        Saves original weights and applies perturbation based on current gradients.
        """
        self._save()
        self._perturb()

    def restore(self):
        """
        Restores the original model weights from backup.
        Should be called after the adversarial forward/backward pass and before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current (clean) model weights.
        Only saves parameters that require gradients and have a gradient computed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _perturb(self):
        """
        Calculates and applies the adversarial perturbation to the weights.
        """
        e = 1e-6  # Small epsilon to prevent division by zero

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0 and not torch.isnan(norm):
                    # Calculate perturbation: direction * step_size
                    # Direction is gradient / norm
                    perturbation = self.adv_lr * grad / (norm + e)

                    # Apply perturbation to the weights
                    param.data.add_(perturbation)

                    # Project weights back onto the epsilon ball centered at original weights
                    # param = min(max(param, orig - eps), orig + eps)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )
