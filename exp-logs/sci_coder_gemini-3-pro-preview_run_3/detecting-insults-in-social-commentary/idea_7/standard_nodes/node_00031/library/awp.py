import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.

    This class implements the AWP strategy to improve model robustness and generalization
    by perturbing weights in the direction of the gradient ascent on the loss surface.
    It targets parameters matching a specific name pattern (default "weight") and
    scales perturbations relative to the weight magnitude.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
    ):
        """
        Initialize the AWP class.

        Args:
            model (torch.nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name pattern to target (default: "weight").
            adv_lr (float): The magnitude of the attack step (learning rate).
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (float): The epoch to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}

    def _save(self):
        """
        Backs up the current model weights before perturbation.
        Only backs up parameters that will be attacked (require grad, have grad, match pattern).
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def attack(self):
        """
        Performs the adversarial attack.
        1. Saves current weights.
        2. Calculates perturbation based on gradients (gradient ascent).
        3. Applies perturbation to weights.
        4. Clips perturbation to be within epsilon bounds relative to original weights.
        """
        self._save()
        e = 1e-6

        for name, param in self.model.named_parameters():
            # Target specific parameters (usually weights, excluding biases if adv_param="weight")
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                grad = param.grad
                data = param.data

                norm_grad = torch.norm(grad)
                norm_data = torch.norm(data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Compute perturbation:
                    # Direction: grad / norm_grad
                    # Scale: adv_lr * norm_data (scale invariant)
                    perturbation = (
                        self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Clip perturbation if epsilon is set
                    # We constrain the new weight to be within [orig - eps*|orig|, orig + eps*|orig|]
                    if self.adv_eps > 0:
                        orig = self.backup[name]
                        eps_val = self.adv_eps * orig.abs()

                        param.data = torch.min(
                            torch.max(param.data, orig - eps_val), orig + eps_val
                        )

    def restore(self):
        """
        Restores the original weights from the backup.
        Should be called after the adversarial forward/backward pass and before the optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to free memory
        self.backup = {}
