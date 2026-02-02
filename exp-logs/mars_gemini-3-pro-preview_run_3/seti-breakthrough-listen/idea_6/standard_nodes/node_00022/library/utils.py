import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array or list): Ground truth binary labels.
        y_pred (np.array or list): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
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


def mixup_data(on_input, off_input, target, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to the dual-stream inputs (On-Target and Off-Target).
    Ensures that the same mixing coefficient (lambda) and permutation index
    are applied to both streams to maintain consistency.

    Args:
        on_input (torch.Tensor): Batch of 'On-Target' inputs.
        off_input (torch.Tensor): Batch of 'Off-Target' inputs.
        target (torch.Tensor): Batch of targets.
        alpha (float): Mixup alpha parameter.
        device (str or torch.device): Device to perform calculations on.

    Returns:
        mixed_on (torch.Tensor): Mixed 'On-Target' inputs.
        mixed_off (torch.Tensor): Mixed 'Off-Target' inputs.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = on_input.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_on = lam * on_input + (1 - lam) * on_input[index, :]
    mixed_off = lam * off_input + (1 - lam) * off_input[index, :]

    y_a, y_b = target, target[index]

    return mixed_on, mixed_off, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
