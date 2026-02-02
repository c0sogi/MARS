import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Jaccard score (intersection over union) between two strings.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
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


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient to maximize loss,
    improving model robustness/generalization.
    """

    def __init__(self, model, optimizer, adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Backs up the current model weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def attack(self):
        """
        Applies adversarial perturbation to the weights.
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # Calculate perturbation: scale gradient by weight magnitude
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    # Add perturbation (Gradient Ascent)
                    param.data.add_(r_at)
                    # Clamp perturbation within epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original model weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
