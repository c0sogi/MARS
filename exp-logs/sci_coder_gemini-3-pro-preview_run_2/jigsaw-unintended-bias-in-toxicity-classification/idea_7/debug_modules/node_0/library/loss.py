import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridContrastiveLoss(nn.Module):
    """
    Implements a hybrid loss function combining:
    1. Binary Cross Entropy (BCE) for the main toxicity target.
    2. BCE for auxiliary identity targets.
    3. Batch-Contrastive Ranking Loss to optimize BPSN and BNSP metrics directly.
    """

    def __init__(self, aux_weight=None, rank_weight=None, rank_margin=None):
        super(HybridContrastiveLoss, self).__init__()
        # Use Config values if overrides are not provided
        self.aux_weight = aux_weight if aux_weight is not None else Config.LAMBDA_AUX
        self.rank_weight = (
            rank_weight if rank_weight is not None else Config.LAMBDA_RANK
        )
        self.rank_margin = (
            rank_margin if rank_margin is not None else Config.RANK_MARGIN
        )

        self.bce = nn.BCEWithLogitsLoss()
        # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
        # We want x1 > x2, so y=1. Loss = max(0, x2 - x1 + margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=self.rank_margin)

    def forward(self, toxicity_logits, identity_logits, targets, identity_labels):
        """
        Args:
            toxicity_logits: (Batch, 1)
            identity_logits: (Batch, Num_Identities)
            targets: (Batch,) or (Batch, 1) - Fractional toxicity scores
            identity_labels: (Batch, Num_Identities) - Fractional identity scores

        Returns:
            total_loss: Scalar tensor
        """
        # Ensure targets have correct shape for BCE
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # 1. Main Toxicity Loss
        loss_tox = self.bce(toxicity_logits, targets)

        # 2. Auxiliary Identity Loss
        loss_aux = self.bce(identity_logits, identity_labels)

        # 3. Batch-Contrastive Ranking Loss
        # We need to binarize targets and identities to define the groups
        # Standard Jigsaw threshold is 0.5
        device = toxicity_logits.device

        targets_bin = (targets >= 0.5).float()
        identities_bin = (identity_labels >= 0.5).float()

        # Flatten toxicity logits for easier indexing
        tox_scores = toxicity_logits.view(-1)

        total_rank_loss = torch.tensor(0.0, device=device)
        pairs_count = 0

        # Iterate over each identity to find specific hard pairs
        num_identities = identity_labels.shape[1]

        for i in range(num_identities):
            # Extract column for current identity
            # Shape: (Batch, 1) -> (Batch,)
            curr_id_mask = identities_bin[:, i]

            # Define masks for the 4 quadrants
            # Toxic (1)
            is_toxic = targets_bin.view(-1) == 1.0
            is_nontoxic = targets_bin.view(-1) == 0.0

            # Identity Present (1)
            has_id = curr_id_mask == 1.0
            no_id = curr_id_mask == 0.0

            # --- BPSN (Background Positive, Subgroup Negative) ---
            # Model confuses Non-Toxic w/ Identity (Neg) with Toxic w/o Identity (Pos)
            # We want: Score(Toxic_NoID) > Score(NonToxic_ID) + margin

            mask_bpsn_pos = is_toxic & no_id  # Toxic, No Identity
            mask_bpsn_neg = is_nontoxic & has_id  # Non-Toxic, Has Identity

            loss_bpsn = self._compute_ranking_loss_for_subset(
                tox_scores, mask_bpsn_pos, mask_bpsn_neg, device
            )

            # --- BNSP (Background Negative, Subgroup Positive) ---
            # Model confuses Toxic w/ Identity (Pos) with Non-Toxic w/o Identity (Neg)
            # We want: Score(Toxic_ID) > Score(NonToxic_NoID) + margin

            mask_bnsp_pos = is_toxic & has_id  # Toxic, Has Identity
            mask_bnsp_neg = is_nontoxic & no_id  # Non-Toxic, No Identity

            loss_bnsp = self._compute_ranking_loss_for_subset(
                tox_scores, mask_bnsp_pos, mask_bnsp_neg, device
            )

            # Accumulate
            if loss_bpsn is not None:
                total_rank_loss += loss_bpsn
                pairs_count += 1

            if loss_bnsp is not None:
                total_rank_loss += loss_bnsp
                pairs_count += 1

        # Normalize ranking loss by the number of identity tasks that contributed
        if pairs_count > 0:
            loss_rank = total_rank_loss / pairs_count
        else:
            loss_rank = torch.tensor(0.0, device=device)

        # Combine Losses
        total_loss = (
            loss_tox + (self.aux_weight * loss_aux) + (self.rank_weight * loss_rank)
        )

        return total_loss

    def _compute_ranking_loss_for_subset(self, scores, mask_pos, mask_neg, device):
        """
        Helper to compute margin ranking loss for all pairs between positive group and negative group.
        """
        # Get indices
        idx_pos = torch.nonzero(mask_pos).squeeze()
        idx_neg = torch.nonzero(mask_neg).squeeze()

        # Handle cases where squeeze returns 0-d tensor or empty
        if idx_pos.numel() == 0 or idx_neg.numel() == 0:
            return None

        # Ensure 1D for meshgrid
        if idx_pos.dim() == 0:
            idx_pos = idx_pos.unsqueeze(0)
        if idx_neg.dim() == 0:
            idx_neg = idx_neg.unsqueeze(0)

        # Create all pairs (Cartesian product)
        # grid_pos: indices of positive samples repeated
        # grid_neg: indices of negative samples repeated
        grid_pos, grid_neg = torch.meshgrid(idx_pos, idx_neg, indexing="ij")

        # Flatten to (N_pairs,)
        flat_pos = grid_pos.flatten()
        flat_neg = grid_neg.flatten()

        # Gather scores
        scores_pos = scores[flat_pos]
        scores_neg = scores[flat_neg]

        # Target for MarginRankingLoss is 1 (meaning input1 should be > input2)
        target_ones = torch.ones_like(scores_pos, device=device)

        return self.ranking_loss(scores_pos, scores_neg, target_ones)
