import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class JigsawLoss(nn.Module):
    """
    Composite loss function for Toxicity Classification with Bias Mitigation.
    Combines:
    1. Weighted Binary Cross Entropy (Primary Toxicity)
    2. Pairwise Margin Ranking Loss (Bias Mitigation)
    3. Auxiliary Multi-label BCE (Regularization via Identity/Subtype heads)
    """

    def __init__(self, ranking_margin=0.5):
        super(JigsawLoss, self).__init__()
        self.ranking_margin = ranking_margin

        # Hyperparameters from Config
        self.lambda_rank = Config.lambda_rank
        self.lambda_aux = Config.lambda_aux

        # Base loss functions
        # We use reduction='none' for the primary loss to apply sample weights manually
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_mean = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        toxicity_logits,
        toxicity_targets,
        aux_logits,
        aux_targets,
        sample_weights=None,
    ):
        """
        Computes the combined loss.

        Args:
            toxicity_logits (torch.Tensor): Logits from the primary toxicity head. Shape (B, 1).
            toxicity_targets (torch.Tensor): Continuous toxicity labels (0.0 to 1.0). Shape (B, 1).
            aux_logits (torch.Tensor): Logits from auxiliary heads (Identities + Subtypes). Shape (B, N_aux).
            aux_targets (torch.Tensor): Multi-label targets for auxiliary tasks. Shape (B, N_aux).
            sample_weights (torch.Tensor, optional): Weights for each sample in the batch. Shape (B, 1) or (B,).

        Returns:
            torch.Tensor: The scalar combined loss.
            dict: A dictionary containing the individual loss components for logging.
        """

        # ==========================================
        # 1. Weighted Pointwise Loss (Primary)
        # ==========================================
        # Calculate element-wise BCE
        # toxicity_targets are fractional; BCEWithLogitsLoss handles this correctly (soft labels)
        bce_loss_elements = self.bce_none(toxicity_logits, toxicity_targets)

        # Apply sample weights if provided
        if sample_weights is not None:
            # Ensure weights are shaped correctly for broadcasting (B, 1)
            if sample_weights.dim() == 1:
                sample_weights = sample_weights.view(-1, 1)

            weighted_bce = bce_loss_elements * sample_weights
            loss_toxicity = weighted_bce.mean()
        else:
            loss_toxicity = bce_loss_elements.mean()

        # ==========================================
        # 2. Pairwise Ranking Loss
        # ==========================================
        # Objective: Penalize if Score(Positive) < Score(Negative) + Margin
        # Especially critical for (Toxic) vs (Non-Toxic + Identity) pairs.

        # Define binary ground truth for ranking (Threshold >= 0.5 is Positive)
        # We detach targets to ensure we don't backprop through label generation logic
        targets_bin = (toxicity_targets >= 0.5).float().detach()

        # Create a mask for valid pairs (i, j) where i is Positive and j is Negative
        # Shape: (B, 1) * (1, B) -> (B, B)
        # valid_pair_mask[i, j] = 1 if (Target_i=1 AND Target_j=0)
        valid_pair_mask = targets_bin.matmul((1 - targets_bin).T)

        # Calculate difference matrix of logits: P_i - P_j
        # Shape: (B, 1) - (1, B) -> (B, B)
        # diff_matrix[i, j] = Logit_i - Logit_j
        diff_matrix = toxicity_logits - toxicity_logits.T

        # Calculate Margin Loss: max(0, margin - (P_i - P_j))
        # We want P_i > P_j + margin, so loss is positive if P_i - P_j < margin
        ranking_loss_matrix = F.relu(self.ranking_margin - diff_matrix)

        # Apply mask to keep only valid (Pos, Neg) pairs
        masked_ranking_loss = ranking_loss_matrix * valid_pair_mask

        # Average over the number of valid pairs
        num_valid_pairs = valid_pair_mask.sum()
        if num_valid_pairs > 0:
            loss_rank = masked_ranking_loss.sum() / num_valid_pairs
        else:
            # Fallback if batch contains only one class
            loss_rank = torch.tensor(0.0, device=toxicity_logits.device)

        # ==========================================
        # 3. Auxiliary Loss (Identities + Subtypes)
        # ==========================================
        # Standard multi-label BCE for the auxiliary heads
        loss_aux = self.bce_mean(aux_logits, aux_targets)

        # ==========================================
        # 4. Combine Losses
        # ==========================================
        total_loss = (
            loss_toxicity
            + (self.lambda_rank * loss_rank)
            + (self.lambda_aux * loss_aux)
        )

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_toxicity": loss_toxicity.item(),
            "loss_rank": loss_rank.item(),
            "loss_aux": loss_aux.item(),
        }
