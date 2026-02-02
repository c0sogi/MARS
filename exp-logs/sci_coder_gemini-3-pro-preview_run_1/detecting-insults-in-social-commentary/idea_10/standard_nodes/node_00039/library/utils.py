import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
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
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted probabilities.

    Returns:
        float: The AUC score.
    """
    return roc_auc_score(y_true, y_pred)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model: PyTorch model instance.
        path: Destination file path.
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=None):
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model: PyTorch model instance.
        path: Source file path.
        device: Device to load the model onto (default: Config.DEVICE).

    Returns:
        model: The model with loaded weights.
    """
    if device is None:
        device = Config.DEVICE

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient to smooth the loss landscape.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=None,
        adv_eps=None,
        start_epoch=None,
    ):
        """
        Args:
            model: The PyTorch model.
            optimizer: The optimizer.
            adv_param: Substring to identify parameters to perturb (default: "weight").
            adv_lr: Step size for the perturbation (default: Config.AWP_LR).
            adv_eps: Maximum perturbation norm (default: Config.AWP_EPS).
            start_epoch: Epoch to start applying AWP (default: Config.AWP_START_EPOCH).
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr if adv_lr is not None else Config.AWP_LR
        self.adv_eps = adv_eps if adv_eps is not None else Config.AWP_EPS
        self.start_epoch = (
            start_epoch if start_epoch is not None else Config.AWP_START_EPOCH
        )
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Internal method to backup current weights and calculate perturbation constraints.
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

    def _restore(self):
        """
        Internal method to restore original weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Backs up the current weights and applies the adversarial perturbation.
        Should be called after the first backward pass.
        """
        self._save()
        self._attack_step()

    def restore(self):
        """
        Restores the original weights.
        Should be called after the adversarial backward pass and before optimizer.step().
        """
        self._restore()

    def _attack_step(self):
        """
        Calculates and applies the perturbation.
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
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    # Constraint perturbation within epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
