import torch


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.

    This class manages the injection of adversarial perturbations into model weights
    during training to flatten the loss landscape and improve generalization.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        """
        Initialize the AWP handler.

        Args:
            model (torch.nn.Module): The PyTorch model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            adv_param (str): Keyword to identify parameters to perturb (default: "weight").
            adv_lr (float): The learning rate (step size) for the adversarial perturbation.
            adv_eps (float): The epsilon bound for the perturbation relative to weight magnitude.
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.

        1. Backs up current weights.
        2. Calculates perturbation direction based on gradients.
        3. Injects perturbation and clips to epsilon bounds.
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Compute norms
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: r_at = alpha * (grad / |grad|) * |weight|
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)

                    # Inject perturbation
                    param.data.add_(r_at)

                    # Clip weights to be within [w - eps*|w|, w + eps*|w|]
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        """
        Backs up the original model weights and computes the clipping bounds.
        Only parameters matching `adv_param` with valid gradients are processed.
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

    def restore(self):
        """
        Restores the original model weights from the backup and clears the cache.
        Should be called after the adversarial backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear backups to free memory and reset for next step
        self.backup = {}
        self.backup_eps = {}
