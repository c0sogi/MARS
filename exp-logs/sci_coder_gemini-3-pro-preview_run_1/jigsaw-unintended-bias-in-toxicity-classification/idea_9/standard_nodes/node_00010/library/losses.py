import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CFG


class HybridLoss(nn.Module):
    """
    Implements the Hybrid Loss function for Toxicity Classification with Bias Mitigation.

    Equation:
    L_total = L_BCE_Weighted + (lambda_rank * L_Rank) + (lambda_aux * L_Aux)

    Components:
    1. L_BCE_Weighted: Binary Cross Entropy on the main toxicity target, weighted by
       bias-centric sample weights to prioritize difficult subgroups.
    2. L_Rank: Pairwise Margin Ranking Loss to ensure toxic examples are ranked
       higher than non-toxic examples (optimizing ROC-AUC).
    3. L_Aux: Binary Cross Entropy on auxiliary identity and attack targets to
       enforce semantic disentanglement in the encoder.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()

        # Main Toxicity Loss
        # reduction='none' is required to apply sample-specific weights manually
        self.main_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Auxiliary Loss
        # Standard BCE for multi-label classification of identities and subtypes
        self.aux_criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # Hyperparameters from Config
        self.rank_weight = CFG.rank_weight
        self.aux_weight = CFG.aux_weight

        # Margin for Ranking Loss (in logit space)
        # A margin of 1.0 is standard for hinge-like losses on logits
        self.margin = 1.0

    def forward(self, outputs, targets, aux_labels, sample_weights):
        """
        Computes the composite loss.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits': Main toxicity logits (batch_size, 1)
                - 'aux_identity': Identity attribute logits (batch_size, 9)
                - 'aux_attack': Identity attack logits (batch_size, 1)
            targets (torch.Tensor): Main toxicity labels (batch_size,)
            aux_labels (torch.Tensor): Concatenated identity and attack labels (batch_size, 10)
            sample_weights (torch.Tensor): Bias-centric weights for the main loss (batch_size,)

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # ====================================================
        # 1. Weighted Pointwise Loss (Main Task)
        # ====================================================
        logits = outputs["logits"]

        # Reshape targets and weights to match logits (batch_size, 1)
        targets_view = targets.view(-1, 1)
        weights_view = sample_weights.view(-1, 1)

        # Compute element-wise BCE
        bce_loss = self.main_criterion(logits, targets_view)

        # Apply sample weights and average
        weighted_bce_loss = (bce_loss * weights_view).mean()

        # ====================================================
        # 2. Auxiliary Loss (Multi-Task Learning)
        # ====================================================
        # Concatenate auxiliary heads to match aux_labels shape
        # aux_identity: (B, 9), aux_attack: (B, 1) -> aux_logits: (B, 10)
        aux_logits = torch.cat([outputs["aux_identity"], outputs["aux_attack"]], dim=1)

        # Compute standard BCE for auxiliary tasks
        aux_loss = self.aux_criterion(aux_logits, aux_labels)

        # ====================================================
        # 3. Pairwise Margin Ranking Loss
        # ====================================================
        # We mine hard pairs within the batch.
        # The batch is sampled to contain "Bias Traps" (e.g., Toxic Background vs Non-Toxic Identity).
        # We enforce: Score(Toxic) > Score(Non-Toxic) + Margin

        # Binarize targets for partitioning (Threshold 0.5 as per task definition)
        binary_targets = (targets >= 0.5).float()

        # Identify indices for Positive (Toxic) and Negative (Non-Toxic) samples
        pos_mask = binary_targets == 1
        neg_mask = binary_targets == 0

        logits_pos = logits[pos_mask]
        logits_neg = logits[neg_mask]

        # Only compute ranking loss if the batch contains at least one of each class
        if logits_pos.size(0) > 0 and logits_neg.size(0) > 0:
            # Broadcast subtract to get matrix of all pairwise differences
            # (P, 1) - (1, N) -> (P, N) matrix where cell (i, j) is score_pos[i] - score_neg[j]
            diff = logits_pos - logits_neg.t()

            # Hinge Loss: max(0, margin - (pos - neg))
            # We want (pos - neg) > margin
            rank_loss = F.relu(self.margin - diff).mean()
        else:
            # Fallback if batch is pure (all toxic or all non-toxic)
            rank_loss = torch.tensor(0.0, device=logits.device)

        # ====================================================
        # Final Combination
        # ====================================================
        total_loss = (
            weighted_bce_loss
            + (self.rank_weight * rank_loss)
            + (self.aux_weight * aux_loss)
        )

        return total_loss
