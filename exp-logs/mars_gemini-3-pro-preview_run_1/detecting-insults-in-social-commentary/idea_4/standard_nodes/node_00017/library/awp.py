import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient ascent to maximize loss,
    regularizing the decision boundary.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    ):
        """
        Initializes the AWP class.

        Args:
            model (torch.nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_lr (float): The learning rate (step size) for the adversarial attack.
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (int): The epoch to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        Saves original weights and applies perturbation based on gradients.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # We apply AWP to weights that have gradients.
            # We typically exclude LayerNorm and Bias terms to maintain training stability.
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                if "weight" in name and "LayerNorm" not in name:
                    # Save the original weight data
                    self.backup[name] = param.data.clone()

                    # Calculate the norm of the gradient
                    grad_norm = torch.norm(param.grad)

                    # Avoid division by zero
                    if grad_norm != 0 and not torch.isnan(grad_norm):
                        # Calculate perturbation: direction * step_size
                        # Direction = grad / ||grad||
                        perturbation = self.adv_lr * param.grad / (grad_norm + e)

                        # Apply perturbation to the weights (Gradient Ascent)
                        param.data.add_(perturbation)

                        # Project the perturbed weights back onto the epsilon ball centered at original weights
                        # This ensures we don't change the weights too drastically
                        param.data = torch.max(
                            torch.min(param.data, self.backup[name] + self.adv_eps),
                            self.backup[name] - self.adv_eps,
                        )

    def restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear the backup to free memory
        self.backup = {}

    def should_attack(self, epoch):
        """
        Determines if AWP should be applied based on the current epoch.

        Args:
            epoch (int): The current training epoch (0-indexed).

        Returns:
            bool: True if AWP should be active, False otherwise.
        """
        return epoch >= self.start_epoch
