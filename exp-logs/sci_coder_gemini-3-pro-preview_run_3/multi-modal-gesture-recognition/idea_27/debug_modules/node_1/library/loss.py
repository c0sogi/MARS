import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TruncatedMSELoss(nn.Module):
    """
    Truncated Mean Squared Error Loss.
    Used for log-space temporal smoothing of predictions.
    Formula: mean(min((log_p[t] - log_p[t-1])^2, threshold^2))
    """

    def __init__(self, threshold=1.0):
        super(TruncatedMSELoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
                              Expected to be log-probabilities.
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure input is 3D (Batch, Channels, Time)
        if x.dim() != 3:
            raise ValueError(
                f"TruncatedMSELoss expects input of shape (B, C, T), got {x.shape}"
            )

        # Compute temporal differences: x[:, :, t] - x[:, :, t-1]
        diff = x[..., 1:] - x[..., :-1]

        # Squared Error
        loss = diff.pow(2)

        # Truncate (Clamp) gradients/values at the threshold
        loss = torch.clamp(loss, max=self.threshold_sq)

        # Mean reduction
        return loss.mean()


class CascadedLoss(nn.Module):
    """
    Cascaded Loss function for the GHC-KRN architecture.
    Combines Weighted Cross-Entropy for all stages and Truncated MSE (Smoothing)
    for refinement stages to implement Deep Supervision with smoothness constraints.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Weighted Cross Entropy
        # Weights: 0.2 for background (index 0), 1.0 for all gesture classes
        self.class_weights = Config.get_class_weights()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # Smoothing Loss (Truncated MSE on Log-Probs)
        self.smooth_loss = TruncatedMSELoss(threshold=Config.SMOOTHING_THRESHOLD)
        self.smooth_weight = Config.WEIGHT_SMOOTHING

    def forward(self, predictions, targets):
        """
        Args:
            predictions (list[torch.Tensor]): List of model outputs (logits) from each stage.
                Expected order: [Stage1_Logits, Stage2_Logits, Stage3_Logits, ...]
                Each tensor shape: (B, C, T) or (B, T, C).
            targets (torch.Tensor): Ground truth labels of shape (B, T).

        Returns:
            torch.Tensor: Total combined loss.
        """
        total_loss = 0.0

        for i, pred in enumerate(predictions):
            # 1. Standardize Input Shape to (B, C, T) for CrossEntropyLoss
            # If shape is (B, T, C), permute it to (B, C, T)
            if (
                pred.shape[1] != Config.NUM_CLASSES
                and pred.shape[-1] == Config.NUM_CLASSES
            ):
                pred = pred.permute(0, 2, 1)

            # 2. Cross Entropy Loss (Applied to all stages for Deep Supervision)
            # pred: (B, C, T), targets: (B, T)
            ce_loss = self.ce_loss(pred, targets)

            stage_loss = ce_loss

            # 3. Smoothing Loss (Only for Refinement Stages, i.e., Stage 2+)
            # Stage 1 (i=0) is the Encoder, which is allowed to be noisy/reactive.
            # Subsequent stages (i>0) are TCNs designed to smooth the output.
            if i > 0:
                # Convert logits to log-probabilities for smoothing
                log_probs = F.log_softmax(pred, dim=1)

                # Calculate smoothing loss
                smooth_loss = self.smooth_loss(log_probs)

                # Add weighted smoothing loss
                stage_loss = stage_loss + (self.smooth_weight * smooth_loss)

            total_loss += stage_loss

        return total_loss
