import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.

    This class implements the AWP technique to improve model robustness and generalization.
    It perturbs the model weights in the direction of the gradient ascent (maximizing loss)
    during training.

    Args:
        model (torch.nn.Module): The model to attack.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        adv_param (str): The name of the parameter to attack (default: "weight").
        adv_lr (float): The magnitude of the perturbation step (learning rate for the attack).
        adv_eps (float): The maximum allowed perturbation norm (epsilon ball).
        start_epoch (int): The epoch to start applying AWP.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}

    def _save(self):
        """
        Saves the current weights of the model parameters that will be perturbed.
        Only saves parameters that have gradients and match the adv_param name.
        """
        for name, param in self.model.named_parameters():
            if (
                param.grad is not None
                and param.requires_grad
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.

        1. Saves the current weights.
        2. Calculates the perturbation direction based on the gradient.
        3. Applies the perturbation to the weights.
        4. Projects the weights to ensure the perturbation is within adv_eps.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.grad is None
                or not param.requires_grad
                or self.adv_param not in name
            ):
                continue

            grad = param.grad
            # Calculate the norm of the gradient
            norm = torch.norm(grad)

            if norm > e and not torch.isnan(norm):
                # Calculate perturbation: direction * step_size
                # We add the gradient because we want to maximize the loss (Gradient Ascent)
                r_at = self.adv_lr * grad / (norm + e)

                # Apply perturbation
                param.data.add_(r_at)

                # Project back to epsilon ball around original weight if epsilon is set
                if self.adv_eps > 0:
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

        # Clear the backup to free memory
        self.backup = {}
