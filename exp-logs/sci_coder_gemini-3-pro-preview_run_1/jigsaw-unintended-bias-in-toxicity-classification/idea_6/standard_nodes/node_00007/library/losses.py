import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridBiasLoss(nn.Module):
    """
    A composite loss function for Toxicity Classification that combines:
    1. Weighted Binary Cross Entropy for the primary toxicity task.
    2. Auxiliary Multi-task Loss for identity and attack prediction.
    3. Pairwise Margin Ranking Loss to explicitly optimize BPSN/BNSP ordering.
    """

    def __init__(self):
        super(HybridBiasLoss, self).__init__()

        # Hyperparameters from Config
        self.lambda_rank = Config.LAMBDA_RANK
        self.lambda_aux = Config.LAMBDA_AUX
        self.ranking_margin = Config.RANKING_MARGIN

        # Primary Loss: Weighted BCE
        # reduction='none' allows us to apply sample weights manually
        self.bce_primary = nn.BCEWithLogitsLoss(reduction="none")

        # Auxiliary Losses: Standard BCE
        self.bce_aux = nn.BCEWithLogitsLoss(reduction="mean")

        # Ranking Loss
        # margin indicates how much higher the positive score should be compared to negative
        self.ranking_loss = nn.MarginRankingLoss(
            margin=self.ranking_margin, reduction="mean"
        )

    def forward(
        self, outputs, targets, sample_weights, aux_identities, aux_identity_attack
    ):
        """
        Computes the combined loss.

        Args:
            outputs (dict): Output dictionary from BiasAwareDeberta model.
                            Contains 'logits', 'aux_identity_logits', 'aux_attack_logits'.
            targets (torch.Tensor): Primary toxicity targets (Batch,).
            sample_weights (torch.Tensor): Weights for each sample (Batch,).
            aux_identities (torch.Tensor): Identity targets (Batch, Num_Identities).
            aux_identity_attack (torch.Tensor): Identity attack targets (Batch,).

        Returns:
            torch.Tensor: The scalar combined loss.
        """
        # ==========================================
        # 1. Primary Toxicity Loss (Weighted)
        # ==========================================
        logits = outputs["logits"].view(-1)
        primary_loss_per_sample = self.bce_primary(logits, targets)

        # Apply sample weights (prioritizing bias traps)
        weighted_primary_loss = (primary_loss_per_sample * sample_weights).mean()

        # ==========================================
        # 2. Auxiliary Heads Loss
        # ==========================================
        # Identity Head (Multi-label)
        identity_logits = outputs["aux_identity_logits"]
        loss_identity = self.bce_aux(identity_logits, aux_identities)

        # Identity Attack Head (Binary)
        attack_logits = outputs["aux_attack_logits"].view(-1)
        loss_attack = self.bce_aux(attack_logits, aux_identity_attack)

        total_aux_loss = loss_identity + loss_attack

        # ==========================================
        # 3. Bias-Aware Ranking Loss
        # ==========================================
        # We use probabilities for ranking to align with the margin scale (0.5)
        probs = torch.sigmoid(logits)

        # Define Masks for Subgroups
        # Thresholds: Target >= 0.5 is Toxic. Identity >= 0.5 is Mentioned.
        is_toxic = targets >= 0.5
        is_nontoxic = ~is_toxic

        # Check if ANY identity is mentioned in the row
        has_identity = (aux_identities >= 0.5).any(dim=1)
        no_identity = ~has_identity

        # Define the 4 quadrants for Bias Analysis
        # 1. Background Positive (Toxic + No Identity) -> Should be High
        mask_bpsn_pos = is_toxic & no_identity

        # 2. Subgroup Negative (Non-Toxic + Identity) -> Should be Low (Model often fails here)
        mask_bpsn_neg = is_nontoxic & has_identity

        # 3. Background Negative (Non-Toxic + No Identity) -> Should be Low
        mask_bnsp_neg = is_nontoxic & no_identity

        # 4. Subgroup Positive (Toxic + Identity) -> Should be High (Model often fails here)
        mask_bnsp_pos = is_toxic & has_identity

        # Extract scores
        scores_bpsn_pos = probs[mask_bpsn_pos]
        scores_bpsn_neg = probs[mask_bpsn_neg]
        scores_bnsp_pos = probs[mask_bnsp_pos]
        scores_bnsp_neg = probs[mask_bnsp_neg]

        loss_rank = torch.tensor(0.0, device=probs.device)

        # --- BPSN Ranking (Toxic No-Ident > Non-Toxic Ident) ---
        # If we have examples in both groups, compute pairwise loss
        if scores_bpsn_pos.size(0) > 0 and scores_bpsn_neg.size(0) > 0:
            # Create all pairs using broadcasting
            # (N, 1) vs (1, M) -> (N, M)
            pos_grid = (
                scores_bpsn_pos.unsqueeze(1)
                .expand(-1, scores_bpsn_neg.size(0))
                .reshape(-1)
            )
            neg_grid = (
                scores_bpsn_neg.unsqueeze(0)
                .expand(scores_bpsn_pos.size(0), -1)
                .reshape(-1)
            )

            # Target is 1 (pos input should be ranked higher than neg input)
            target_ones = torch.ones_like(pos_grid)
            loss_rank += self.ranking_loss(pos_grid, neg_grid, target_ones)

        # --- BNSP Ranking (Toxic Ident > Non-Toxic No-Ident) ---
        if scores_bnsp_pos.size(0) > 0 and scores_bnsp_neg.size(0) > 0:
            pos_grid = (
                scores_bnsp_pos.unsqueeze(1)
                .expand(-1, scores_bnsp_neg.size(0))
                .reshape(-1)
            )
            neg_grid = (
                scores_bnsp_neg.unsqueeze(0)
                .expand(scores_bnsp_pos.size(0), -1)
                .reshape(-1)
            )

            target_ones = torch.ones_like(pos_grid)
            loss_rank += self.ranking_loss(pos_grid, neg_grid, target_ones)

        # ==========================================
        # 4. Combine
        # ==========================================
        total_loss = (
            weighted_primary_loss
            + (self.lambda_aux * total_aux_loss)
            + (self.lambda_rank * loss_rank)
        )

        return total_loss
