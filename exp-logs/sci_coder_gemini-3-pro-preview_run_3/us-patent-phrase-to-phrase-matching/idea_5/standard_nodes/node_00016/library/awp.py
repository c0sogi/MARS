import torch
from library.config import Config
from library.utils import get_logger

logger = get_logger("awp")


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to flatten the loss landscape,
    improving generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
        adv_param="weight",
    ):
        """
        Initialize the AWP attacker.

        Args:
            model (torch.nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            adv_lr (float): The step size (learning rate) for the adversarial perturbation.
            adv_eps (float): The maximum magnitude (epsilon) of the perturbation.
            start_epoch (int): The epoch number to start applying AWP.
            adv_param (str): Substring to filter parameters to attack (default: "weight").
                             This typically excludes biases.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_param = adv_param
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack step.
        1. Identifies relevant parameters (those with gradients and matching `adv_param`).
        2. Backs up the original parameter values.
        3. Computes the perturbation based on the gradient direction and weight magnitude.
        4. Applies the perturbation to the parameters, projected within the `adv_eps` ball.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            # Check if parameter should be attacked:
            # 1. Requires grad
            # 2. Has a computed gradient
            # 3. Name matches the filter (e.g., contains "weight")
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                # Save the original data if not already saved in this step
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                # Compute and apply perturbation if gradient is valid
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation:
                    # Direction: grad / norm_grad
                    # Magnitude: adv_lr * norm_data (adaptive to weight scale)
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Update weight
                    param.data.add_(r_at)

                    # Project (clip) the weight to ensure it stays within epsilon range of the original
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original weights of the model from the backup.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to prepare for the next iteration
        self.backup = {}
