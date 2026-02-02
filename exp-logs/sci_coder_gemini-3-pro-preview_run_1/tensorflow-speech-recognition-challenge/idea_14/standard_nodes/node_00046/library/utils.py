import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Performs Mixup augmentation on the batch.

    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution.
        device (torch.device): Device to store the indices.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Interpolation coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss for Mixup inputs.

    Args:
        criterion (callable): Loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Interpolation coefficient.

    Returns:
        loss (torch.Tensor): Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DistillationLoss(nn.Module):
    """
    Loss function for Self-Distillation (Born-Again Networks).
    Computes a weighted sum of Cross-Entropy Loss (Student vs Truth)
    and KL-Divergence Loss (Student vs Teacher).

    Formula: L = lambda * L_CE + (1 - lambda) * L_KL
    """

    def __init__(self, distillation_weight=0.5, temperature=1.0):
        """
        Args:
            distillation_weight (float): Weight (lambda) for the Cross-Entropy part.
                                         (1 - lambda) is applied to the KL part.
            temperature (float): Temperature scaling for Softmax/LogSoftmax.
        """
        super(DistillationLoss, self).__init__()
        self.distillation_weight = distillation_weight
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets, mixup_args=None):
        """
        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model.
            targets (torch.Tensor): Ground truth labels.
            mixup_args (tuple, optional): Tuple of (y_a, y_b, lam) if mixup is used.

        Returns:
            total_loss (torch.Tensor): Combined loss.
        """
        # 1. Cross-Entropy Loss (Student vs Truth)
        if mixup_args is not None:
            y_a, y_b, lam = mixup_args
            loss_ce = mixup_criterion(self.ce_loss, student_logits, y_a, y_b, lam)
        else:
            loss_ce = self.ce_loss(student_logits, targets)

        # 2. KL Divergence Loss (Student vs Teacher)
        # Apply temperature scaling
        # Student: Log-Softmax
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Teacher: Softmax (Targets for KL)
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Calculate KL Divergence
        # Note: Standard distillation often scales gradients by T^2, but we adhere
        # strictly to the weighted sum formula provided in the task description.
        loss_kl = self.kl_div(student_log_probs, teacher_probs)

        # Combine losses
        total_loss = (self.distillation_weight * loss_ce) + (
            (1 - self.distillation_weight) * loss_kl
        )

        return total_loss
