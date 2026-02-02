import torch


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of gradient ascent to maximize loss,
    smoothing the loss landscape and improving generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1e-4,
        adv_eps=1e-2,
    ):
        """
        Initialize AWP.

        Args:
            model (torch.nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): Keyword to filter parameters to perturb (default: "weight").
                             Usually excludes biases and layer norms.
            adv_lr (float): The step size (learning rate) for the adversarial attack.
                            Controls the magnitude of perturbation relative to weight norm.
            adv_eps (float): The maximum allowed perturbation (epsilon constraint).
                             (Note: In single-step AWP with small adv_lr, this acts as a bound).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.
        Saves original weights and applies perturbation based on gradient direction.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            # Apply perturbation only to parameters that:
            # 1. Require gradients
            # 2. Have computed gradients
            # 3. Match the filter (e.g., are weights, not biases)
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):

                # Compute norms
                # Note: Gradient scaling (AMP) cancels out in the normalization step
                grad_norm = torch.norm(param.grad)
                weight_norm = torch.norm(param.data.detach())

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation:
                    # Direction = grad / ||grad||
                    # Magnitude = adv_lr * ||weight||
                    # We use weight_norm to make the perturbation relative to parameter scale
                    r_at = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )

                    # Apply perturbation to the weights
                    param.data.add_(r_at)

    def _save(self):
        """
        Backs up the current weights of the model before perturbation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    # Clone the data to ensure we have a detached copy
                    self.backup[name] = param.data.clone()

    def restore(self):
        """
        Restores the original weights from the backup and clears the backup storage.
        Should be called after the adversarial forward/backward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        # Clear the backup to reset for the next iteration and save memory
        self.backup = {}
