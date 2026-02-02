import torch
import torch.nn as nn
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.

    This class implements the AWP technique to improve model robustness and generalization.
    It works by injecting worst-case perturbations into the model weights (ascending the loss surface)
    during training, forcing the model to find a flatter minimum.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = "weight",
        adv_lr: float = Config.AWP_LR,
        adv_eps: float = Config.AWP_EPS,
        start_epoch: float = Config.AWP_START_EPOCH,
    ):
        """
        Initialize AWP.

        Args:
            model (nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The learning rate for the adversarial step (perturbation magnitude).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (float): The epoch number to start applying AWP.
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
        Saves the current values of the parameters to be perturbed.
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

    def _restore(self):
        """
        Restores the parameters to their original values from the backup.
        Clears the backup after restoration.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}

    def attack_step(self):
        """
        Performs the adversarial attack step.

        1. Backs up the current model weights.
        2. Calculates the perturbation for each targeted parameter based on its gradient.
           The perturbation is in the direction of the gradient (ascent) to maximize loss.
        3. Scales the perturbation by the parameter's norm and adv_lr.
        4. Clips the perturbation to ensure it doesn't exceed adv_eps (relative to param norm).
        5. Updates the parameter weights with the perturbation.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Retrieve gradient and current weight data
                grad = param.grad
                data = param.data

                # Calculate norms
                grad_norm = torch.norm(grad)
                data_norm = torch.norm(data)

                if grad_norm == 0:
                    continue

                # Calculate the perturbation
                # Direction: grad / grad_norm (Gradient Ascent direction)
                # Scale: adv_lr * data_norm (Step size relative to weight magnitude)
                perturbation = (grad / (grad_norm + e)) * (self.adv_lr * data_norm)

                # Enforce Epsilon Constraint (Projection)
                # We ensure the perturbation magnitude does not exceed adv_eps * data_norm
                perturbation_norm = torch.norm(perturbation)
                limit = self.adv_eps * data_norm

                if perturbation_norm > limit:
                    # Scale down the perturbation to fit within the limit
                    perturbation = perturbation * (limit / (perturbation_norm + e))

                # Apply the perturbation to the weights
                # We add the perturbation because we want to ascend the loss surface (maximize loss)
                param.data.add_(perturbation)
