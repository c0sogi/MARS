import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Temporal Mean Squared Error (T-MSE) Loss in Probability Space.
    Cite Lesson 33: Prefer Probability-Space over Log-Space for Temporal Smoothing.

    This loss encourages temporal smoothness by penalizing frame-to-frame variations
    in the predicted softmax probabilities.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (Batch, Classes, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to probabilities (Softmax) - Cite Lesson 33
        probs = F.softmax(logits, dim=1)

        # Compute temporal differences: P_t - P_{t-1}
        diff = probs[:, :, 1:] - probs[:, :, :-1]

        # Square the differences (MSE)
        # No truncation needed for bounded probabilities [0, 1]
        loss = diff.pow(2).mean()

        return loss


class MultiStageLoss(nn.Module):
    """
    Multi-Stage Loss for the IC-RCN Architecture.

    Aggregates losses from the Generation stage (Bi-LSTM) and two Refinement stages (TCNs).
    - All stages use Weighted Cross-Entropy Loss to handle class imbalance (Background vs Gestures).
    - Refinement stages additionally use T-MSE Loss to smooth predictions.

    Total Loss = L_gen + L_ref1 + L_ref2
    """

    def __init__(self):
        super(MultiStageLoss, self).__init__()

        # Initialize Class Weights
        # 0.1 for Background, 1.0 for Gestures
        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float).to(
            Config.DEVICE
        )

        # Weighted Cross-Entropy Loss
        # Expects inputs: (Batch, Classes, Time), Targets: (Batch, Time)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        # T-MSE Loss for smoothing refinement stages
        self.tmse_loss = TMSELoss()
        self.tmse_weight = Config.TMSE_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs for each stage.
                            Keys: 'gen', 'ref1', 'ref2'.
                            Values: Tensors of shape (Batch, Classes, Time).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Time).

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (torch.Tensor): The scalar loss for backpropagation.
                - loss_dict (dict): Breakdown of loss components for logging.
        """
        # ---------------------------------------------------------------------
        # 1. Generation Stage (Bi-LSTM)
        # ---------------------------------------------------------------------
        # Only Cross-Entropy is applied here to allow the LSTM to learn dynamics
        # without excessive smoothing constraints initially.
        gen_logits = outputs["gen"]
        loss_gen = self.ce_loss(gen_logits, targets)

        # ---------------------------------------------------------------------
        # 2. Refinement Stage 1 (Dilated TCN)
        # ---------------------------------------------------------------------
        # Applies CE for classification and T-MSE for coarse smoothing.
        ref1_logits = outputs["ref1"]
        loss_ref1_ce = self.ce_loss(ref1_logits, targets)
        loss_ref1_tmse = self.tmse_loss(ref1_logits)
        loss_ref1 = loss_ref1_ce + (self.tmse_weight * loss_ref1_tmse)

        # ---------------------------------------------------------------------
        # 3. Refinement Stage 2 (Dilated TCN)
        # ---------------------------------------------------------------------
        # Applies CE and T-MSE for fine-grained boundary refinement.
        ref2_logits = outputs["ref2"]
        loss_ref2_ce = self.ce_loss(ref2_logits, targets)
        loss_ref2_tmse = self.tmse_loss(ref2_logits)
        loss_ref2 = loss_ref2_ce + (self.tmse_weight * loss_ref2_tmse)

        # ---------------------------------------------------------------------
        # Aggregation
        # ---------------------------------------------------------------------
        total_loss = loss_gen + loss_ref1 + loss_ref2

        # Dictionary for monitoring
        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_gen": loss_gen.item(),
            "loss_ref1_total": loss_ref1.item(),
            "loss_ref1_ce": loss_ref1_ce.item(),
            "loss_ref1_tmse": loss_ref1_tmse.item(),
            "loss_ref2_total": loss_ref2.item(),
            "loss_ref2_ce": loss_ref2_ce.item(),
            "loss_ref2_tmse": loss_ref2_tmse.item(),
        }

        return total_loss, loss_dict
