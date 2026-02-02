import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The average ROC AUC score across all columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Only calculate AUC if the class contains both 0 and 1
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                pass

    if not scores:
        return 0.5

    return np.mean(scores)


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights in the direction of the gradient ascent to maximize loss,
    encouraging the model to find flatter, more robust minima.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.2,
        start_epoch=0,
        scaler=None,
    ):
        """
        Args:
            model: The PyTorch model.
            optimizer: The optimizer.
            adv_param (str): The parameter name substring to target (default: "weight").
            adv_lr (float): The magnitude of the perturbation step.
            adv_eps (float): The maximum allowed perturbation (epsilon).
            start_epoch (int): The epoch to start applying AWP.
            scaler: Optional GradScaler for mixed precision.
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
        Backs up the current weights of the targeted parameters and calculates
        the clipping constraints based on epsilon.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

                    # Calculate constraints: original_weight +/- (eps * |original_weight|)
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

        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack step:
        1. Saves current weights.
        2. Computes perturbation based on gradients (Gradient Ascent).
        3. Applies perturbation to weights, clipped to epsilon constraints.
        """
        self._save()
        e = 1e-6

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = param.grad

                norm1 = torch.norm(grad)
                norm2 = torch.norm(param.data.detach())

                if norm1 != 0 and not torch.isnan(norm1):
                    # Compute perturbation: direction * step_size * magnitude_scaling
                    r_at = self.adv_lr * grad / (norm1 + e) * (norm2 + e)

                    # Apply perturbation
                    param.data.add_(r_at)

                    # Clip weights to stay within the epsilon ball of the original weights
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
