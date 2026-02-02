import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss for temporal smoothing of probability distributions.
    Computes the squared difference between adjacent frames, clamped to a maximum threshold.

    This encourages the model to produce smooth probability transitions without penalizing
    sharp, legitimate changes that exceed the threshold (truncation).
    """

    def __init__(self, threshold=0.15):
        """
        Args:
            threshold (float): The maximum allowed difference value before truncation.
                               The squared error is clamped to threshold^2.
        """
        super(TMSELoss, self).__init__()
        self.threshold = threshold**2

    def forward(self, probs, mask):
        """
        Args:
            probs (torch.Tensor): Softmax probabilities of shape (Batch, Time, Classes).
            mask (torch.Tensor): Sequence mask of shape (Batch, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate difference between adjacent frames: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared difference
        sq_diff = diff**2

        # Truncate (clamp) the squared errors to prevent outlier domination
        truncated_sq_diff = torch.clamp(sq_diff, min=0, max=self.threshold)

        # Create mask for valid transitions (both t and t-1 must be valid)
        # Shape: (B, T-1)
        mask_valid = mask[:, 1:] * mask[:, :-1]

        # Expand mask for classes: (B, T-1, 1)
        mask_valid_expanded = mask_valid.unsqueeze(2)

        # Compute masked mean
        # Sum over all dims and divide by number of valid elements * channels
        numerator = torch.sum(truncated_sq_diff * mask_valid_expanded)
        denominator = torch.sum(mask_valid_expanded) * probs.shape[2]

        # Avoid division by zero
        loss = numerator / (denominator + 1e-6)

        return loss


class DCSGCNLoss(nn.Module):
    """
    Composite loss function for the DCSGCN model.
    Aggregates Classification, Boundary, and Smoothing losses across all stages
    (Deep Supervision) with specific weighting.
    """

    def __init__(self):
        super(DCSGCNLoss, self).__init__()

        # Hyperparameters for loss weighting
        self.w_cls = config.HYPERPARAMS["w_cls"]
        self.w_bnd = config.HYPERPARAMS["w_bnd"]
        self.w_smooth = config.HYPERPARAMS["w_smooth"]

        # Class Weights for handling imbalance (Background vs Gestures)
        # Register as buffer to automatically handle device placement (CPU/GPU)
        weight_tensor = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("class_weights", weight_tensor)

        # Sub-losses
        # We use NLLLoss because the model outputs Softmax probabilities.
        # Taking log(probs) + NLLLoss is mathematically equivalent to CrossEntropy on logits.
        self.criterion_cls = nn.NLLLoss(weight=self.class_weights, reduction="none")
        self.criterion_bnd = nn.BCELoss(reduction="none")
        self.criterion_smooth = TMSELoss(threshold=config.HYPERPARAMS["tmse_threshold"])

    def forward(self, model_outputs, labels, boundaries, mask):
        """
        Computes the total weighted loss across all model stages.

        Args:
            model_outputs (dict): Dictionary containing outputs for 'stage1', 'stage2', 'stage3'.
                                  Each value is a tuple (cls_probs, bnd_probs).
            labels (torch.Tensor): Ground truth labels (Batch, Time).
            boundaries (torch.Tensor): Ground truth boundaries (Batch, Time).
            mask (torch.Tensor): Sequence mask (Batch, Time).

        Returns:
            torch.Tensor: Total aggregated loss.
        """
        total_loss = 0.0

        # Iterate over all stages for Deep Supervision
        stages = ["stage1", "stage2", "stage3"]

        for stage in stages:
            if stage not in model_outputs:
                continue

            cls_probs, bnd_probs = model_outputs[stage]

            # --- Classification Loss ---
            # Inputs: cls_probs (B, T, C), labels (B, T)
            # Flatten for loss computation: (B*T, C) and (B*T)
            B, T, C = cls_probs.shape

            # Add epsilon for numerical stability before log
            log_probs = torch.log(cls_probs + 1e-8)

            cls_flat = log_probs.reshape(-1, C)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            # Compute element-wise NLL loss
            loss_cls_elem = self.criterion_cls(cls_flat, labels_flat)

            # Apply mask and compute mean over valid frames
            loss_cls = torch.sum(loss_cls_elem * mask_flat) / (
                torch.sum(mask_flat) + 1e-6
            )

            # --- Boundary Loss ---
            # Inputs: bnd_probs (B, T, 1) -> squeeze to (B, T)
            bnd_squeezed = bnd_probs.squeeze(2)

            loss_bnd_elem = self.criterion_bnd(bnd_squeezed, boundaries)
            loss_bnd = torch.sum(loss_bnd_elem * mask) / (torch.sum(mask) + 1e-6)

            # --- Smoothing Loss ---
            # Inputs: cls_probs (B, T, C) - TMSE works on probabilities directly
            loss_smooth = self.criterion_smooth(cls_probs, mask)

            # --- Aggregation ---
            stage_loss = (
                (self.w_cls * loss_cls)
                + (self.w_bnd * loss_bnd)
                + (self.w_smooth * loss_smooth)
            )

            total_loss += stage_loss

        return total_loss
