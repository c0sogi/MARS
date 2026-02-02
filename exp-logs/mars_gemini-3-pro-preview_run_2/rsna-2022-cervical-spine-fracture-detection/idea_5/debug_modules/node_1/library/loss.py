import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCEFocalLoss(nn.Module):
    """
    Binary Cross Entropy Focal Loss for multi-label classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the class with label y=1.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - Raw scores from the model (before sigmoid).
            targets: (batch_size, num_classes) - Binary ground truth labels (0 or 1).
        """
        # Compute binary cross entropy loss per element
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate p_t: probability associated with the true class
        # If target=1, p_t = p; if target=0, p_t = 1-p
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Calculate alpha_t: balancing factor
        # If target=1, alpha_t = alpha; if target=0, alpha_t = 1-alpha
        # Note: In some implementations alpha is only applied to class 1.
        # Here we use the balanced form often cited in object detection.
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Calculate Focal Loss
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class HybridLoss(nn.Module):
    """
    Combined loss function for the 2.5D Anatomically-Guided Attention Network.

    Components:
    1. Main Task (Fracture Detection): Focal Loss
       - Handles class imbalance (fractures are rare).
       - Predicts 7 vertebrae + 1 overall label.

    2. Auxiliary Task (Anatomical Localization): Cross Entropy Loss
       - Supervised by 'Silver Standard' segmentation/bounding box data.
       - Predicts anatomical level (C1-C7, Background) per slice.
       - Masked to ignore slices without ground truth.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.config = Config()

        # Main Task Loss
        self.main_loss_fn = BCEFocalLoss(
            alpha=self.config.FOCAL_ALPHA, gamma=self.config.FOCAL_GAMMA
        )

        # Auxiliary Task Loss
        # ignore_index handles slices where we don't have anatomical labels
        self.aux_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.AUX_LOSS_IGNORE_INDEX, reduction="mean"
        )

        self.aux_weight = self.config.AUX_LOSS_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'fracture_logits': (batch_size, num_classes)
                - 'aux_logits': (batch_size, seq_len, aux_num_classes)

            targets (dict): Dictionary containing ground truth:
                - 'fracture_labels': (batch_size, num_classes)
                - 'aux_labels': (batch_size, seq_len) - Integer class indices

        Returns:
            dict: Dictionary containing 'loss' (total), 'main_loss', and 'aux_loss'.
        """
        fracture_logits = outputs["fracture_logits"]
        aux_logits = outputs["aux_logits"]

        fracture_labels = targets["fracture_labels"]
        aux_labels = targets["aux_labels"]

        # --- 1. Main Task Loss (Focal) ---
        # Ensure targets are float for BCE
        loss_main = self.main_loss_fn(fracture_logits, fracture_labels.float())

        # --- 2. Auxiliary Task Loss (Cross Entropy) ---
        # Reshape for CrossEntropyLoss:
        # Input: (N, C, d1...) -> Here (Batch * Seq, Classes)
        # Target: (N, d1...) -> Here (Batch * Seq)

        batch_size, seq_len, num_aux_classes = aux_logits.shape

        # Flatten batch and sequence dimensions
        aux_logits_flat = aux_logits.view(-1, num_aux_classes)
        aux_labels_flat = aux_labels.view(-1)

        loss_aux = self.aux_loss_fn(aux_logits_flat, aux_labels_flat)

        # --- 3. Combine ---
        total_loss = loss_main + (self.aux_weight * loss_aux)

        return {"loss": total_loss, "main_loss": loss_main, "aux_loss": loss_aux}
