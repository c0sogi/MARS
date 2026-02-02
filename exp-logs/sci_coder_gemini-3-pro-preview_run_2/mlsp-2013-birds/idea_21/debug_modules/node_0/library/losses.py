import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Logits and Positive Class Weighting.

    This loss function is designed to handle class imbalance by assigning
    higher weights to the positive class (presence of a bird species).
    It wraps torch.nn.BCEWithLogitsLoss.
    """

    def __init__(self, pos_weights=None):
        """
        Args:
            pos_weights (torch.Tensor or np.ndarray, optional):
                Weights for the positive class for each label.
                Shape should be (num_classes,).
                If None, no weighting is applied (weights=1.0).
        """
        super(WeightedBCELoss, self).__init__()

        self.pos_weights = pos_weights

        # Initialize the criterion.
        # Note: pos_weight must be set to the correct device in forward() or init
        # if device is known. We handle device placement in forward() for flexibility.
        if self.pos_weights is not None:
            if not isinstance(self.pos_weights, torch.Tensor):
                self.pos_weights = torch.tensor(self.pos_weights, dtype=torch.float32)
            # Register as buffer so it's part of state_dict but not a parameter
            self.register_buffer("weight_tensor", self.pos_weights)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth binary labels of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for BCEWithLogitsLoss
        targets = targets.float()

        # If weights exist, ensure the internal criterion uses the weights on the correct device
        if hasattr(self, "weight_tensor"):
            if self.criterion.pos_weight.device != logits.device:
                self.criterion.pos_weight = self.weight_tensor.to(logits.device)

        return self.criterion(logits, targets)


class DistillationLoss(nn.Module):
    """
    Composite loss function for Anchor-Based Self-Distillation.

    Combines:
    1. Weighted BCE Loss (Hard Targets): Learns from ground truth labels.
    2. KL Divergence Loss (Soft Targets): Learns from Anchor Model logits (Teacher).

    Formula:
    L = BCE_weighted(Student, Truth) + lambda * (T^2) * KLDiv(Student_Soft, Teacher_Soft)
    """

    def __init__(
        self,
        pos_weights=None,
        lambda_param=Config.DISTILLATION_LAMBDA,
        temperature=Config.DISTILLATION_TEMP,
    ):
        """
        Args:
            pos_weights (torch.Tensor, optional): Class weights for the hard BCE loss.
            lambda_param (float): Weighting factor for the distillation (soft) loss component.
            temperature (float): Temperature T to soften probability distributions.
        """
        super(DistillationLoss, self).__init__()

        self.lambda_param = lambda_param
        self.temperature = temperature

        # Hard Loss Component
        self.hard_loss_fn = WeightedBCELoss(pos_weights=pos_weights)

        # Soft Loss Component
        # KLDivLoss expects input in log-probabilities
        self.kl_div_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits (torch.Tensor): Logits from the model being trained (Student).
            teacher_logits (torch.Tensor): Logits from the pre-trained Anchor models (Teacher).
            targets (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Combined scalar loss.
        """
        # 1. Calculate Hard Loss (Student vs Ground Truth)
        loss_hard = self.hard_loss_fn(student_logits, targets)

        # 2. Calculate Soft Loss (Student vs Teacher)
        # Apply temperature scaling
        # Student: LogSoftmax(logits / T)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Teacher: Softmax(logits / T)
        # We detach teacher logits to ensure no gradients flow back to the teacher
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Calculate KL Divergence
        # We scale by T^2 to keep gradients invariant to the magnitude of T
        loss_soft = self.kl_div_fn(student_log_probs, teacher_probs) * (
            self.temperature**2
        )

        # 3. Combine
        total_loss = loss_hard + (self.lambda_param * loss_soft)

        return total_loss


def calculate_pos_weights(labels):
    """
    Utility function to calculate positive class weights based on label frequency.

    Formula: weight_i = (Total_Samples - Pos_Samples_i) / Pos_Samples_i

    Args:
        labels (np.ndarray or torch.Tensor): Binary label matrix of shape (N, num_classes).

    Returns:
        torch.Tensor: Calculated weights of shape (num_classes,).
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_samples = len(labels)
    pos_counts = np.sum(labels, axis=0)

    # Avoid division by zero by clipping counts to at least 1
    pos_counts = np.clip(pos_counts, 1, num_samples)

    neg_counts = num_samples - pos_counts
    weights = neg_counts / pos_counts

    return torch.tensor(weights, dtype=torch.float32)
