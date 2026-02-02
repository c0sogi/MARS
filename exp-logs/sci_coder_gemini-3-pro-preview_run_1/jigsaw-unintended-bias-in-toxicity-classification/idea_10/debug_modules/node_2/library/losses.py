import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CFG


class JigsawLoss(nn.Module):
    """
    Composite loss function for Jigsaw Toxicity Classification.
    Combines Weighted BCE for toxicity, BCE for auxiliary heads, and
    Margin Ranking Loss for robust metric optimization.
    """

    def __init__(self):
        super().__init__()
        # Load weights from configuration
        self.toxicity_weight = CFG.toxicity_loss_weight
        self.aux_identity_weight = CFG.aux_identity_loss_weight
        self.aux_attack_weight = CFG.aux_attack_loss_weight
        self.ranking_weight = CFG.ranking_loss_weight

        # Margin for ranking loss (enforcing separation between Toxic-Background and NonToxic-Identity)
        self.ranking_margin = 0.5

        # Primary loss: reduction='none' allows applying sample-specific weights (Bias Trap weighting)
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")

        # Auxiliary losses: standard mean reduction
        self.bce_mean = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, outputs, targets, stage="general"):
        """
        Compute the composite loss.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits': (batch_size, 1) - Primary toxicity score
                - 'aux_identity_logits': (batch_size, num_identities)
                - 'aux_attack_logits': (batch_size, 1)
            targets (dict): Dictionary containing ground truth:
                - 'target': (batch_size, 1) - Toxicity label
                - 'identities': (batch_size, num_identities) - Identity labels
                - 'attack': (batch_size, 1) - Identity attack label
                - 'sample_weights': (batch_size, 1) - Weights for bias traps
            stage (str): Current training stage ('dapt', 'general', 'robust').
                         Ranking loss is only applied in 'robust' stage.

        Returns:
            tuple: (total_loss, loss_dict)
        """
        # ====================================================
        # 1. Primary Toxicity Loss (Weighted BCE)
        # ====================================================
        logits = outputs["logits"]
        target = targets["target"]
        weights = targets.get("sample_weights", torch.ones_like(target))

        # Compute element-wise BCE
        loss_tox_raw = self.bce_none(logits, target)
        # Apply bias-centric sample weights and average
        loss_tox = (loss_tox_raw * weights).mean()

        # ====================================================
        # 2. Auxiliary Identity Loss (Multi-label BCE)
        # ====================================================
        aux_id_logits = outputs["aux_identity_logits"]
        aux_id_targets = targets["identities"]
        loss_aux_id = self.bce_mean(aux_id_logits, aux_id_targets)

        # ====================================================
        # 3. Auxiliary Identity Attack Loss (BCE)
        # ====================================================
        aux_att_logits = outputs["aux_attack_logits"]
        aux_att_targets = targets["attack"]
        loss_aux_att = self.bce_mean(aux_att_logits, aux_att_targets)

        # ====================================================
        # 4. Robust Ranking Loss (Conditional)
        # ====================================================
        loss_ranking = torch.tensor(0.0, device=logits.device)

        if stage == "robust":
            # Binarize targets for masking (Threshold 0.5)
            is_toxic = (target >= 0.5).float()
            is_nontoxic = (target < 0.5).float()

            # Identity presence (Threshold 0.5)
            identity_presence = (aux_id_targets >= 0.5).float()

            num_identities = identity_presence.shape[1]
            total_pairs = 0
            accumulated_margin_loss = 0.0

            # Iterate over each identity to mine BPSN pairs
            # BPSN: Background Positive (Toxic, No Identity) vs Subgroup Negative (Non-Toxic, Has Identity)
            # Goal: Score(Toxic Background) > Score(Non-Toxic Identity)
            for k in range(num_identities):
                # Masks for current identity k
                has_id_k = identity_presence[:, k].unsqueeze(1)  # (B, 1)
                no_id_k = 1.0 - has_id_k

                # Identify candidates
                # Toxic Background: Toxic AND NOT Identity k
                mask_tb = (is_toxic * no_id_k).bool().squeeze(1)

                # Non-Toxic Identity: Non-Toxic AND Identity k
                mask_nti = (is_nontoxic * has_id_k).bool().squeeze(1)

                # Extract logits
                logits_tb = logits[mask_tb]
                logits_nti = logits[mask_nti]

                # If we have candidates for both sides, compute pairwise loss
                if logits_tb.numel() > 0 and logits_nti.numel() > 0:
                    # Broadcast to form all pairs: (N_tb, 1) - (1, N_nti) -> (N_tb, N_nti)
                    diff = logits_tb.unsqueeze(1) - logits_nti.unsqueeze(0)

                    # We want diff > 0 (TB > NTI).
                    # Loss = ReLU(margin - diff)
                    pair_losses = F.relu(self.ranking_margin - diff)

                    accumulated_margin_loss += pair_losses.mean()
                    total_pairs += 1

            # Average ranking loss over the number of identities that had valid pairs
            if total_pairs > 0:
                loss_ranking = accumulated_margin_loss / total_pairs

        # ====================================================
        # 5. Combine Losses
        # ====================================================
        total_loss = (
            self.toxicity_weight * loss_tox
            + self.aux_identity_weight * loss_aux_id
            + self.aux_attack_weight * loss_aux_att
            + self.ranking_weight * loss_ranking
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_tox": loss_tox.item(),
            "loss_aux_id": loss_aux_id.item(),
            "loss_aux_att": loss_aux_att.item(),
            "loss_ranking": loss_ranking.item(),
        }

        return total_loss, loss_dict
