import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient ascent to maximize loss,
    improving model robustness and generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    ):
        """
        Args:
            model (torch.nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): The name of the parameters to attack (default: "weight").
            adv_lr (float): The magnitude of the attack step.
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch to start applying AWP.
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
        Saves the current weights of the parameters that will be attacked.
        Only saves parameters that require gradients and match the adv_param pattern.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _attack(self):
        """
        Perturbs the model weights based on the gradient direction.
        Should be called after loss.backward() to use the computed gradients.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0 and not torch.isnan(norm):
                    # Calculate perturbation: alpha * grad / ||grad||
                    # We add a small epsilon 'e' to avoid division by zero
                    r_at = self.adv_lr * grad / (norm + e)

                    # Apply perturbation to the weights
                    param.data.add_(r_at)

                    # Project back to epsilon ball if epsilon is set
                    # This ensures the weights don't drift too far from the original
                    if self.adv_eps > 0:
                        orig = self.backup[name]
                        min_value = orig - self.adv_eps
                        max_value = orig + self.adv_eps
                        param.data = torch.max(
                            torch.min(param.data, max_value), min_value
                        )

    def _restore(self):
        """
        Restores the original model weights from the backup.
        Should be called after the adversarial forward/backward pass and before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory and ensure fresh backup next time
        self.backup = {}
