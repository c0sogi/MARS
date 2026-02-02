import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Temporal Mean Squared Error Loss for probability smoothing.
    Computes the Mean Squared Error between probabilities at time t and t-1.

    As per instructions:
    - Applied to Softmax probabilities.
    - Unclamped (standard MSE on differences).
    - Not conditioned on boundary predictions.
    - Strictly masked.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) Softmax probabilities.
            mask: (B, T) Boolean mask indicating valid frames.
        Returns:
            loss: Scalar tensor.
        """
        # Calculate temporal differences: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Determine valid transitions based on mask
        # We need both t and t-1 to be valid
        # Shape: (B, T-1)
        valid_transitions = mask[:, 1:] & mask[:, :-1]

        # If no valid transitions exist (e.g., sequence length < 2), return 0
        if not valid_transitions.any():
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        # Select valid differences
        # valid_diffs shape: (N_valid, C)
        valid_diffs = diff[valid_transitions]

        # Compute MSE: mean(diff^2)
        loss = torch.mean(valid_diffs**2)

        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Multi-stage loss function for DSG-CRCN.
    Aggregates Classification, Boundary, and Smoothing losses across 3 stages.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

        # Load class weights from Config
        # Weights: 0.1 for Background (idx 0), 1.0 for Gestures (idx 1-20)
        self.class_weights = Config.get_class_weights_tensor()

        # Loss components
        # We use NLLLoss because the model outputs Softmax probabilities.
        # We will take torch.log(probs) before passing to NLLLoss.
        self.cls_criterion = nn.NLLLoss(weight=self.class_weights, reduction="mean")

        # Binary Cross Entropy for boundary detection
        self.bnd_criterion = nn.BCELoss(reduction="mean")

        # Smoothing loss
        self.tmse_criterion = TMSELoss()

        # Component weights from Config
        self.w_cls = Config.LOSS_WEIGHT_CLS
        self.w_bnd = Config.LOSS_WEIGHT_BND
        self.w_tmse = Config.LOSS_WEIGHT_TMSE

    def forward(self, outputs, targets, boundaries, mask):
        """
        Args:
            outputs: Dictionary containing model outputs:
                     'stage1_cls', 'stage1_bnd',
                     'stage2_cls', 'stage2_bnd',
                     'stage3_cls', 'stage3_bnd'
                     Shapes: cls -> (B, T, NumClasses), bnd -> (B, T, 1)
            targets: (B, T) LongTensor of class labels.
            boundaries: (B, T) FloatTensor of boundary labels (0 or 1).
            mask: (B, T) BoolTensor of valid frames.

        Returns:
            total_loss: Scalar tensor.
            metrics: Dictionary of individual loss components for logging.
        """
        total_loss = 0.0
        metrics = {}

        # Flatten targets and boundaries based on mask for efficient computation
        # valid_targets: (N_valid,)
        valid_targets = targets[mask]
        # valid_boundaries: (N_valid, 1)
        valid_boundaries = boundaries[mask].unsqueeze(-1)

        # Iterate over stages
        stages = [1, 2, 3]

        for s in stages:
            cls_key = f"stage{s}_cls"
            bnd_key = f"stage{s}_bnd"

            # Retrieve stage outputs
            # probs: (B, T, C)
            cls_probs = outputs[cls_key]
            # bnd_probs: (B, T, 1)
            bnd_probs = outputs[bnd_key]

            # --- 1. Classification Loss ---
            # Apply mask to probabilities
            # valid_cls_probs: (N_valid, C)
            valid_cls_probs = cls_probs[mask]

            # Numerical stability for log
            eps = 1e-8
            valid_log_probs = torch.log(valid_cls_probs + eps)

            loss_cls = self.cls_criterion(valid_log_probs, valid_targets)

            # --- 2. Boundary Loss ---
            # valid_bnd_probs: (N_valid, 1)
            valid_bnd_probs = bnd_probs[mask]

            loss_bnd = self.bnd_criterion(valid_bnd_probs, valid_boundaries)

            # --- 3. Smoothing Loss (TMSE) ---
            # TMSE handles its own masking internally using the temporal structure
            loss_tmse = self.tmse_criterion(cls_probs, mask)

            # --- Aggregate Stage Loss ---
            stage_loss = (
                (self.w_cls * loss_cls)
                + (self.w_bnd * loss_bnd)
                + (self.w_tmse * loss_tmse)
            )

            total_loss += stage_loss

            # Log metrics
            metrics[f"loss_s{s}_cls"] = loss_cls.item()
            metrics[f"loss_s{s}_bnd"] = loss_bnd.item()
            metrics[f"loss_s{s}_tmse"] = loss_tmse.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
