import torch
from library.config import CFG


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights to maximize loss, improving robustness and generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
        scaler=None,
    ):
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name pattern to attack (default: "weight").
            adv_lr (float): The magnitude of the attack step.
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Save the original weights of the parameters that will be attacked.
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
        Restore the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Perform the adversarial attack on the model weights.
        1. Save current weights.
        2. Calculate perturbation based on gradients.
        3. Apply perturbation to weights.
        4. Clip weights to stay within the epsilon ball of the original weights.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                # Retrieve gradient
                grad = param.grad

                # Calculate norms
                grad_norm = torch.norm(grad)
                weight_norm = torch.norm(param.data).detach()

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation:
                    # Direction: grad / grad_norm
                    # Magnitude: adv_lr * weight_norm (Scale relative to weight size)
                    r_at = self.adv_lr * grad / (grad_norm + e) * (weight_norm + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project/Clip to ensure perturbation is within adv_eps
                    # We use a simple absolute clipping around the original value here.
                    if name in self.backup:
                        param.data = torch.min(
                            torch.max(param.data, self.backup[name] - self.adv_eps),
                            self.backup[name] + self.adv_eps,
                        )
