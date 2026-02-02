import torch
import torch.nn as nn


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights to maximize loss, encouraging the model to find
    flatter minima in the loss landscape for better generalization and robustness.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch=0, scaler=None):
        """
        Args:
            model (nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_lr (float): The step size for the adversarial perturbation.
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (int): The epoch to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Scaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current model weights before perturbation.
        Only saves parameters that require gradients and have gradients computed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def restore(self):
        """
        Restores the model weights to the saved state (before perturbation).
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        """
        Perturbs the model weights in the direction of the gradient (gradient ascent).
        Scales perturbation by weight magnitude and clips within epsilon.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: direction * step_size * weight_scale
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clip perturbation to be within epsilon of original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def attack_backward(self, inputs, labels, criterion, epoch):
        """
        Performs the adversarial attack and backward pass.

        1. Saves current weights.
        2. Perturbs weights to maximize loss (attack).
        3. Performs forward pass with perturbed weights.
        4. Calculates adversarial loss.
        5. Performs backward pass to accumulate gradients.
        6. Restores original weights.

        Args:
            inputs (dict/tensor): Inputs to the model.
            labels (tensor): Ground truth labels.
            criterion (callable): Loss function.
            epoch (int): Current training epoch.

        Returns:
            torch.Tensor: The adversarial loss (or None if AWP is not active).
        """
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        self._attack_step()

        try:
            # Handle different input types (dict for transformers, tensor otherwise)
            if isinstance(inputs, dict):
                output = self.model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                output = self.model(*inputs)
            else:
                output = self.model(inputs)

            adv_loss = criterion(output, labels)

            if self.scaler:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

        finally:
            # Always restore weights, even if forward/backward fails
            self.restore()

        return adv_loss
