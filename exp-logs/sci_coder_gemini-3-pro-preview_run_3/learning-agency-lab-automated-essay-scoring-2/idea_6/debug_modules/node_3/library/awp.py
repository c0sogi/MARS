import torch
from library.config import CFG


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights to maximize the loss, improving robustness and generalization.
    """

    def __init__(
        self, model, optimizer=None, adv_param="weight", adv_lr=None, adv_eps=None
    ):
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (optim.Optimizer): The optimizer used for training (optional).
            adv_param (str): The name of the parameters to attack (default: "weight").
            adv_lr (float): The magnitude of the perturbation (learning rate for AWP).
            adv_eps (float): The epsilon value (often used for clamping, though basic AWP relies on lr).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr if adv_lr is not None else CFG.awp_lr
        self.adv_eps = adv_eps if adv_eps is not None else CFG.awp_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Backs up current weights and applies adversarial perturbation.
        """
        self._save()
        self._attack_step()

    def restore(self):
        """
        Restores the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current weights of parameters to be attacked.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    # We could also backup epsilon constraints here if implementing PGD

    def _attack_step(self):
        """
        Calculates and applies the perturbation to the weights.
        Formula: delta = adv_lr * grad / |grad| * |weight|
        """
        e = 1e-6
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
                    # Calculate perturbation scaled by weight magnitude
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)

                    # Apply perturbation
                    param.data.add_(r_at)
