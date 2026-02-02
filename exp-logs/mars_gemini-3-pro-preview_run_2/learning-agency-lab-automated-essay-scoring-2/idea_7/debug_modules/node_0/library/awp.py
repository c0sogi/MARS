import torch
from torch.cuda.amp import autocast


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation for robust training.
    Perturbs model weights in the direction of the gradient ascent to flatten the loss landscape.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch, scaler=None):
        """
        Initialize AWP.

        Args:
            model (torch.nn.Module): The model to attack.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_lr (float): The learning rate (step size) for the adversarial perturbation.
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (int): The epoch at which to start applying AWP.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, input_ids, attention_mask, labels, criterion, epoch):
        """
        Performs the full AWP routine:
        1. Checks if AWP is active for the current epoch.
        2. Saves current weights and applies perturbation (attack_step).
        3. Computes loss and gradients on the perturbed model.
        4. Restores original weights (restore_step).

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Target labels.
            criterion (callable): Loss function.
            epoch (int): Current training epoch.
        """
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return

        # 1. Save weights and apply perturbation
        self.attack_step()

        # 2. Zero gradients to ensure we update using only the adversarial gradients
        # (This strategy replaces clean gradients with adversarial ones)
        self.optimizer.zero_grad()

        # 3. Forward pass with perturbed weights
        with autocast(enabled=True):
            output = self.model(input_ids, attention_mask)
            adv_loss = criterion(output, labels)

        # 4. Backward pass
        if self.scaler:
            self.scaler.scale(adv_loss).backward()
        else:
            adv_loss.backward()

        # 5. Restore original weights
        self.restore_step()

    def attack_step(self):
        """
        Saves the current model weights and applies adversarial perturbation
        based on the existing gradients.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and param.grad.norm() > 0:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Compute perturbation: direction * magnitude
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clamp parameters within epsilon ball around original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def restore_step(self):
        """
        Restores the original model weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Internal method to back up current weights and calculate epsilon constraints.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and param.grad.norm() > 0:
                if name not in self.backup:
                    # Save original data
                    self.backup[name] = param.data.clone()

                    # Calculate min/max bounds based on adv_eps
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )
