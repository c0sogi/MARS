import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) metric.

    Handles conversion from PyTorch tensors to NumPy arrays and ensures
    predictions are rounded to integers and clipped to the [1, 6] range
    as required by the essay scoring rubric.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores (can be floats).

    Returns:
        float: The QWK score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Round and clip predictions to match the 1-6 integer scale
    # This is crucial because QWK is a categorical metric
    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, 1, 6)

    # Calculate QWK
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation for robust training.

    This class perturbs the model weights in the direction that maximizes the loss
    (gradient ascent) to flatten the loss landscape and improve generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1.0,
        adv_eps: float = 0.2,
        start_epoch: int = 0,
        scaler=None,
    ):
        """
        Args:
            model (torch.nn.Module): The model to perturb.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            adv_param (str): The parameter name substring to target (usually "weight").
            adv_lr (float): The learning rate for the adversarial step.
            adv_eps (float): The maximum magnitude of the perturbation (epsilon).
            start_epoch (int): The epoch to start applying AWP.
            scaler: GradScaler for mixed precision training (optional).
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

    def attack_step(self, epoch: int):
        """
        Performs the adversarial attack on the model weights.

        Should be called after the first backward pass. It saves the original weights,
        calculates the perturbation based on gradients, and applies it.

        Args:
            epoch (int): The current training epoch.
        """
        if epoch < self.start_epoch:
            return

        e = 1e-6
        for name, param in self.model.named_parameters():
            # Apply perturbation only to targeted parameters (e.g., weights, not bias)
            # and only if they have gradients.
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Save original parameter
                self.backup[name] = param.data.clone()

                # Calculate gradient norm
                grad = param.grad.data
                norm = torch.norm(grad)

                # Calculate perturbation
                if norm != 0 and not torch.isnan(norm):
                    # Direction * Step Size
                    r_at = self.adv_lr * grad / (norm + e)
                    # Apply perturbation to the parameter
                    param.data.add_(r_at)
                    # Clamp the perturbation to be within epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self, epoch: int):
        """
        Restores the original model weights.

        Should be called after the second backward pass (on perturbed weights)
        and before the optimizer step.

        Args:
            epoch (int): The current training epoch.
        """
        if epoch < self.start_epoch:
            return

        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        self.backup_eps = {}
