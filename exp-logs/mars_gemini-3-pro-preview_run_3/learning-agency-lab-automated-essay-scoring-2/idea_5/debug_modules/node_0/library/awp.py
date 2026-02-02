import torch
import torch.nn as nn


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.

    This class implements the AWP technique to improve model generalization by
    perturbing weights in the direction that maximizes the loss (gradient ascent)
    before the optimization step.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1.0,
        adv_eps: float = 0.2,
        start_epoch: int = 0,
        scaler: torch.amp.GradScaler = None,
    ):
        """
        Initialize AWP.

        Args:
            model (nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name pattern to attack (default: "weight").
            adv_lr (float): The magnitude of the attack (learning rate for perturbation).
            adv_eps (float): The maximum norm constraint for the perturbation.
            start_epoch (int): The epoch to start applying AWP.
            scaler (torch.amp.GradScaler, optional): GradScaler for mixed precision training.
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

    def attack_backward(self, input_ids, attention_mask, labels, criterion, epoch):
        """
        Performs the adversarial attack and the backward pass for the adversarial loss.

        Steps:
        1. Check if AWP should be active (epoch >= start_epoch).
        2. Save current model weights.
        3. Perturb weights to maximize loss.
        4. Forward pass with perturbed weights.
        5. Calculate loss and backward pass (accumulate gradients).
        6. Restore original weights.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Target labels.
            criterion (callable): Loss function.
            epoch (int): Current training epoch.
        """
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return

        # 1. Save original weights
        self._save()

        # 2. Perturb weights
        self._attack_step()

        try:
            # 3. Forward pass with perturbed weights
            # Enable mixed precision if scaler is provided
            # We assume CUDA device as AWP is computationally intensive
            with torch.amp.autocast("cuda", enabled=(self.scaler is not None)):
                y_preds = self.model(input_ids, attention_mask)
                adv_loss = criterion(y_preds, labels)

            # 4. Backward pass
            if self.scaler:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

        finally:
            # 5. Restore original weights
            self._restore()

    def _attack_step(self):
        """
        Perturbs the model weights based on the gradients accumulated from the
        standard forward/backward pass.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: direction * magnitude
                    # Direction = grad / |grad|
                    # Magnitude = adv_lr * |weight|
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Enforce epsilon constraint
                    # Ensure perturbed weight is within [orig - eps, orig + eps]
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        """
        Saves the original weights of the parameters to be attacked and
        calculates the epsilon bounds.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    # Save original weight
                    self.backup[name] = param.data.clone()

                    # Calculate epsilon bounds based on weight magnitude
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        """
        Restores the original weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory
        self.backup = {}
        self.backup_eps = {}
