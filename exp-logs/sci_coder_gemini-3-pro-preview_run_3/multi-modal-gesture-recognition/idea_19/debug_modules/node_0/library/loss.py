import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiTaskRefinementLoss(nn.Module):
    """
    Implements the combined loss for the ASK-RN architecture.
    Components:
    1. Weighted Cross Entropy for Classification (Stages 1, 2, 3)
    2. Binary Cross Entropy for Auxiliary Boundary Detection (Stage 1)
    3. Truncated Log-Space MSE for Temporal Smoothness (Stages 2, 3)
    """

    def __init__(self):
        super(MultiTaskRefinementLoss, self).__init__()

        # 1. Classification Loss
        # Load weights from Config
        class_weights = Config.CLASS_WEIGHTS
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights).float()

        # Initialize CrossEntropyLoss with class weights
        # Note: Weights will be moved to device in forward pass if needed
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

        # 2. Boundary Loss
        # BCEWithLogitsLoss combines Sigmoid and BCE for stability
        self.bnd_criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # 3. Hyperparameters
        self.lambda_bnd = Config.LAMBDA_BOUNDARY
        self.lambda_smooth = Config.LAMBDA_SMOOTH
        self.smooth_threshold = Config.SMOOTH_LOSS_THRESHOLD

    def compute_truncated_smooth_loss(self, logits):
        """
        Computes the Truncated MSE loss in Log-Space.
        Formula: mean( clamp( (log(p_t) - log(p_{t-1}))^2, max=threshold ) )

        Args:
            logits: (Batch, Frames, Classes)
        """
        # Compute Log Probabilities: log(softmax(x))
        log_probs = F.log_softmax(logits, dim=2)

        # Compute temporal differences: log(p_t) - log(p_{t-1})
        # Slice 1: [1, 2, ..., T]
        # Slice 2: [0, 1, ..., T-1]
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        mse = diff.pow(2)

        # Truncate (Clamp) the error
        # This prevents large penalties for sharp, necessary transitions
        truncated_mse = torch.clamp(mse, max=self.smooth_threshold)

        # Average over all elements
        return torch.mean(truncated_mse)

    def forward(self, outputs, targets):
        """
        Calculates the total loss.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits_s1': (Batch, Frames, Num_Classes)
                - 'logits_bnd': (Batch, Frames, 1)
                - 'logits_s2': (Batch, Frames, Num_Classes)
                - 'logits_s3': (Batch, Frames, Num_Classes)
            targets (dict): Dictionary containing ground truth:
                - 'cls_labels': (Batch, Frames) LongTensor
                - 'bnd_labels': (Batch, Frames) FloatTensor (0 or 1)

        Returns:
            total_loss (Tensor): Scalar loss for backpropagation.
            metrics (dict): Dictionary of individual loss components for logging.
        """
        # Unpack outputs
        logits_s1 = outputs["logits_s1"]
        logits_bnd = outputs["logits_bnd"]
        logits_s2 = outputs["logits_s2"]
        logits_s3 = outputs["logits_s3"]

        # Unpack targets
        cls_labels = targets["cls_labels"]
        bnd_labels = targets["bnd_labels"]

        # Device management for CrossEntropy weights
        device = logits_s1.device
        if (
            self.cls_criterion.weight is not None
            and self.cls_criterion.weight.device != device
        ):
            self.cls_criterion.weight = self.cls_criterion.weight.to(device)

        # ---------------------------------------------------------------------
        # 1. Classification Loss (Deep Supervision on S1, S2, S3)
        # ---------------------------------------------------------------------
        # Reshape for CrossEntropy: (N, C) logits, (N) labels
        num_classes = logits_s1.shape[2]

        flat_logits_s1 = logits_s1.reshape(-1, num_classes)
        flat_logits_s2 = logits_s2.reshape(-1, num_classes)
        flat_logits_s3 = logits_s3.reshape(-1, num_classes)
        flat_cls_labels = cls_labels.reshape(-1)

        loss_cls_s1 = self.cls_criterion(flat_logits_s1, flat_cls_labels)
        loss_cls_s2 = self.cls_criterion(flat_logits_s2, flat_cls_labels)
        loss_cls_s3 = self.cls_criterion(flat_logits_s3, flat_cls_labels)

        loss_cls_total = loss_cls_s1 + loss_cls_s2 + loss_cls_s3

        # ---------------------------------------------------------------------
        # 2. Auxiliary Boundary Loss (S1 only)
        # ---------------------------------------------------------------------
        # Reshape for BCE: (N) logits, (N) labels
        flat_logits_bnd = logits_bnd.reshape(-1)
        flat_bnd_labels = bnd_labels.reshape(-1)

        loss_bnd = self.bnd_criterion(flat_logits_bnd, flat_bnd_labels)

        # ---------------------------------------------------------------------
        # 3. Smoothness Loss (Refinement Stages S2, S3)
        # ---------------------------------------------------------------------
        loss_smooth_s2 = self.compute_truncated_smooth_loss(logits_s2)
        loss_smooth_s3 = self.compute_truncated_smooth_loss(logits_s3)

        loss_smooth_total = loss_smooth_s2 + loss_smooth_s3

        # ---------------------------------------------------------------------
        # Total Loss
        # ---------------------------------------------------------------------
        total_loss = (
            loss_cls_total
            + (self.lambda_bnd * loss_bnd)
            + (self.lambda_smooth * loss_smooth_total)
        )

        # Metrics for logging
        metrics = {
            "loss_total": total_loss.item(),
            "loss_cls_s1": loss_cls_s1.item(),
            "loss_cls_s2": loss_cls_s2.item(),
            "loss_cls_s3": loss_cls_s3.item(),
            "loss_bnd": loss_bnd.item(),
            "loss_smooth": loss_smooth_total.item(),
        }

        return total_loss, metrics
