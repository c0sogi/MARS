import torch
from library.config import Config


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
        scaler=None,
    ):
        """
        Adversarial Weight Perturbation (AWP) class.

        Args:
            model: The neural network model to attack.
            optimizer: The optimizer used for training.
            adv_lr: The learning rate for the attack (step size).
            adv_eps: The epsilon value (magnitude of perturbation).
            start_epoch: The epoch to start applying AWP.
            scaler: Optional GradScaler for mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}

    def attack_step(self, epoch):
        """
        Performs the adversarial attack on the model weights.
        This should be called after loss.backward() and before optimizer.step().

        Args:
            epoch (int): The current training epoch.
        """
        if epoch < self.start_epoch:
            return

        self._save()
        e = 1e-6

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                # Calculate norms
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad > 0 and norm_data > 0:
                    # FGM (Fast Gradient Method) perturbation logic
                    # Perturbation is proportional to the weight magnitude and aligned with gradient
                    # delta = eps * (g / ||g||) * ||w||
                    perturbation = (
                        self.adv_eps * grad / (norm_grad + e) * (norm_data + e)
                    )

                    # Apply perturbation to the weights
                    param.data.add_(perturbation)

    def _save(self):
        """
        Saves the current model weights to a backup dictionary.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def restore(self):
        """
        Restores the model weights from the backup dictionary.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backup to save memory
        self.backup = {}
