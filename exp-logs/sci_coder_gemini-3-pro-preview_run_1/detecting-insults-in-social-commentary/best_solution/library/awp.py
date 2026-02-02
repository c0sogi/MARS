import torch
from library.utils import get_logger


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights to maximize loss, creating a smoother loss landscape.
    This helps the model generalize better and be more robust to noise.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch=1, scaler=None):
        """
        Args:
            model (nn.Module): The model to perturb.
            optimizer (Optimizer): The optimizer used for training.
            adv_lr (float): The magnitude of the adversarial step (learning rate).
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
            start_epoch (int): The epoch at which to start applying AWP.
            scaler (GradScaler, optional): PyTorch AMP GradScaler for mixed precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.logger = get_logger()

    def _save(self):
        """
        Saves the current model weights for parameters that have gradients.
        This allows us to restore the model to its original state after the adversarial step.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def _restore(self):
        """
        Restores the saved model weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def attack(self):
        """
        Perturbs the model weights based on gradients (Gradient Ascent).
        The perturbation is proportional to the weight magnitude and gradient direction.
        """
        self._save()
        e = 1e-6  # Small constant to prevent division by zero

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                # Calculate norms
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Compute perturbation: delta = adv_lr * (grad / ||grad||) * ||weight||
                    # We add delta because we want to maximize loss (ascent)
                    perturbation = (
                        self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Clip perturbation to epsilon ball if adv_eps is set
                    # This ensures the weights don't drift too far from the original values
                    if self.adv_eps > 0:
                        min_val = self.backup[name] - self.adv_eps
                        max_val = self.backup[name] + self.adv_eps
                        param.data = torch.max(torch.min(param.data, max_val), min_val)

    def restore(self):
        """
        Public method to restore weights. Should be called after the adversarial backward pass.
        """
        self._restore()

    def step(self, epoch):
        """
        Manages the adversarial training cycle.
        Checks if the current epoch allows for AWP, and if so, initiates the attack.

        Usage in training loop:
            # 1. Normal Forward & Backward
            loss.backward()

            # 2. AWP Attack
            if awp.step(epoch):
                # 3. Adversarial Forward & Backward
                loss_adv = criterion(model(inputs), targets)
                loss_adv.backward()

                # 4. Restore Weights
                awp.restore()

            # 5. Optimizer Step
            optimizer.step()

        Args:
            epoch (int): Current training epoch (1-based).

        Returns:
            bool: True if attack was performed, False otherwise.
        """
        if epoch >= self.start_epoch:
            self.attack()
            return True
        return False
