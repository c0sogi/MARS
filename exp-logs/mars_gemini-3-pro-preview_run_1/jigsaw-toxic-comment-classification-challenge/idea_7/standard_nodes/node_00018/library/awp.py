import torch


class AWP:
    """
    Adversarial Weight Perturbation (AWP) utility.
    Perturbs model weights to maximize loss during training, improving robustness.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
        start_epoch=0,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model to attack.
            optimizer: The optimizer used for training.
            adv_param (str): The parameter name pattern to attack (default: "weight").
            adv_lr (float): The magnitude of the attack (learning rate for the adversary).
            adv_eps (float): The maximum allowed perturbation (epsilon ball).
            start_epoch (int): The epoch to start applying AWP.
            scaler: Optional GradScaler for mixed precision training.
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
        Saves the original weights of the parameters to be attacked.
        Also calculates the epsilon bounds for clipping the perturbation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def attack(self):
        """
        Calculates the perturbation direction based on gradients and modifies
        the model weights to maximize the loss.
        """
        e = 1e-6

        # Ensure weights are saved before attacking
        if not self.backup:
            self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate perturbation direction
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Perturbation formula: lr * grad / |grad| * |weight|
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Project back to epsilon ball (clipping)
                    if name in self.backup_eps:
                        param.data = torch.min(
                            torch.max(param.data, self.backup_eps[name][0]),
                            self.backup_eps[name][1],
                        )

    def _restore(self):
        """
        Restores the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to release memory
        self.backup = {}
        self.backup_eps = {}
