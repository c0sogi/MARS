import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    Formula:
        loss = (1 - epsilon) * CE(targets) + epsilon * CE(uniform_distribution)

    Args:
        epsilon (float): Smoothing factor.
        reduction (str): Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted logits [Batch, Classes, Time] or [Batch, Classes].
            targets (torch.Tensor): Ground truth labels [Batch, Time] or [Batch].
        """
        # Ensure preds are [N, C, ...]
        n_classes = preds.size(1)

        # Calculate Log Softmax
        log_preds = F.log_softmax(preds, dim=1)

        # Compute the loss
        # 1. Loss with respect to the true class (standard CE part)
        # We use nll_loss which expects log_probs
        loss_nll = F.nll_loss(log_preds, targets, reduction=self.reduction)

        # 2. Loss with respect to the uniform distribution (smoothing part)
        # Mean of log_preds over classes gives the negative entropy-like term
        # sum(-1/K * log(p)) = -1/K * sum(log(p))
        loss_smooth = -log_preds.mean(dim=1)

        if self.reduction == "mean":
            loss_smooth = loss_smooth.mean()
        elif self.reduction == "sum":
            loss_smooth = loss_smooth.sum()

        return (1.0 - self.epsilon) * loss_nll + self.epsilon * loss_smooth


class LogSpaceSmoothingLoss(nn.Module):
    """
    Temporal Smoothing Loss operating in Log-Space (Truncated MSE).
    Penalizes rapid fluctuations in prediction confidence between adjacent frames.

    Formula:
        diff = log(P_t) - log(P_{t-1})
        loss = clamp(diff^2, max=threshold)

    Args:
        weight (float): Weighting factor for this loss.
        threshold (float): Maximum penalty value (truncation) to allow genuine transitions.
    """

    def __init__(self, weight: float = 0.15, threshold: float = 1.0):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape [Batch, Classes, Time].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Expecting [Batch, Classes, Time]
        if logits.dim() != 3:
            # If not 3D, we can't compute temporal smoothness easily unless it's [Batch, Time, Classes]
            # Assuming standard PyTorch [N, C, L] format for 1D convolution outputs
            return torch.tensor(0.0, device=logits.device)

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Calculate differences between adjacent frames: P_t - P_{t-1}
        # Slice along time dimension (dim=2)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Error
        mse = diff.pow(2)

        # Truncate (Clamp) the error
        truncated_mse = torch.clamp(mse, max=self.threshold)

        # Mean reduction
        loss = truncated_mse.mean()

        return self.weight * loss


class CascadedLoss(nn.Module):
    """
    Aggregated Loss function for the Three-Stage Cascaded Network.
    Combines Label Smoothing Cross Entropy for all stages (Deep Supervision)
    and Log-Space Smoothing for temporal consistency.

    Total Loss = Sum(L_cls(Stage_i)) + Sum(L_smooth(Stage_i))
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Classification Loss
        self.cls_loss = LabelSmoothingCrossEntropy(
            epsilon=Config.LABEL_SMOOTHING, reduction="mean"
        )

        # Temporal Smoothing Loss
        self.smooth_loss = LogSpaceSmoothingLoss(
            weight=Config.LOG_SMOOTHING_WEIGHT, threshold=Config.LOG_SMOOTHING_THRESHOLD
        )

    def forward(self, outputs, targets):
        """
        Args:
            outputs (tuple): Tuple containing logits from (stage1, stage2, stage3).
                             Each tensor shape: [Batch, Classes, Time]
            targets (torch.Tensor): Ground truth labels. Shape: [Batch, Time]

        Returns:
            torch.Tensor: Total aggregated loss.
            dict: Dictionary containing individual loss components for logging.
        """
        stage1_logits, stage2_logits, stage3_logits = outputs

        # 1. Classification Losses (Deep Supervision)
        l1_cls = self.cls_loss(stage1_logits, targets)
        l2_cls = self.cls_loss(stage2_logits, targets)
        l3_cls = self.cls_loss(stage3_logits, targets)

        total_cls_loss = l1_cls + l2_cls + l3_cls

        # 2. Temporal Smoothing Losses
        # Applied to all stages to encourage smooth features throughout
        l1_smooth = self.smooth_loss(stage1_logits)
        l2_smooth = self.smooth_loss(stage2_logits)
        l3_smooth = self.smooth_loss(stage3_logits)

        total_smooth_loss = l1_smooth + l2_smooth + l3_smooth

        # Total Loss
        total_loss = total_cls_loss + total_smooth_loss

        loss_dict = {
            "loss": total_loss.item(),
            "l1_cls": l1_cls.item(),
            "l2_cls": l2_cls.item(),
            "l3_cls": l3_cls.item(),
            "l_smooth": total_smooth_loss.item(),
        }

        return total_loss, loss_dict
