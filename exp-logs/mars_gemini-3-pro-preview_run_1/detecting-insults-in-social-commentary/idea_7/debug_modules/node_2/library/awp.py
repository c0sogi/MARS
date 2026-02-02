import torch


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.

    This class implements the AWP technique to improve model robustness and generalization
    by perturbing the model weights in the direction of the gradient ascent (maximizing loss)
    during training. It is particularly effective for noisy labels or when seeking to flatten
    the loss landscape.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1e-4,
        adv_eps: float = 1e-4,
    ):
        """
        Initialize the AWP attacker.

        Args:
            model (torch.nn.Module): The PyTorch model to apply AWP to.
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            adv_param (str): The substring to match parameter names against (e.g., "weight").
                             Only parameters containing this string will be perturbed.
            adv_lr (float): The step size (learning rate) for the adversarial perturbation.
            adv_eps (float): Epsilon value for numerical stability or constraints.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model parameters.

        This method iterates through the model's parameters. If a parameter matches the
        `adv_param` criteria and has a valid gradient, it calculates a perturbation
        proportional to the gradient direction and the parameter's magnitude.
        Original parameters are backed up before modification.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Filter parameters: must require grad, have grad computed, and match the name filter
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                # Avoid division by zero or perturbing if gradient is NaN
                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: scale * (grad / |grad|) * |param|
                    # This ensures perturbation is relative to parameter magnitude
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Save original parameter data
                    self.backup[name] = param.data.clone()

                    # Apply perturbation to the parameter
                    param.data.add_(r_at)

    def restore(self):
        """
        Restores the original model parameters from the backup.

        This should be called after the forward and backward pass on the perturbed
        model are complete, to ensure the optimizer step is applied to the original
        (or consistently updated) weights, not the perturbed ones.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}
