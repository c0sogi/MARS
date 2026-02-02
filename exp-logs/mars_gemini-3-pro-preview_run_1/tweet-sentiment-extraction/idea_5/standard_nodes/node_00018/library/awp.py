import torch
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights to maximize loss, improving robustness and generalization.
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
            model (nn.Module): The model to attack.
            optimizer (optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name to target (default: "weight").
            adv_lr (float): The learning rate for the adversarial perturbation.
            adv_eps (float): The epsilon constraint for the perturbation.
            start_epoch (int): The epoch to start applying AWP.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Perturbs the model weights based on the gradients to maximize loss.
        Should be called after loss.backward() and before the second forward pass.
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: lr * grad / |grad| * |weight|
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)

                    # Project back to epsilon ball centered at original weight
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def _save(self):
        """
        Saves the current model weights to a backup dictionary.
        Only saves parameters that are being targeted by AWP.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _restore(self):
        """
        Restores the model weights from the backup dictionary.
        Clears the backup after restoration.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
