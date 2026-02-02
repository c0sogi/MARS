import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridLoss(nn.Module):
    """
    Composite loss function for Toxicity Classification with Bias Mitigation.

    This loss implements the 'Hybrid Pairwise Ranking' strategy by combining:
    1. Weighted Binary Cross Entropy: Prioritizes 'Bias Traps' (difficult subgroups)
       using sample weights calculated in the data pipeline.
    2. Pairwise Margin Ranking Loss: Directly optimizes the ROC-AUC metric by
       enforcing a margin between Toxic and Non-Toxic examples.
    3. Auxiliary Multi-Task Loss: Supervises the identity and attack type heads
       to learn robust, disentangled representations.
    """

    def __init__(self):
        super().__init__()
        # BCE with reduction='none' allows manual application of sample weights
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")
        # Standard BCE for auxiliary tasks
        self.bce_mean = nn.BCEWithLogitsLoss(reduction="mean")

        # Hyperparameters from Config
        self.margin = Config.RANKING_MARGIN
        self.alpha = Config.ALPHA_RANK
        self.beta = Config.BETA_AUX

        # Number of identity columns to correctly split the auxiliary target tensor
        self.num_identities = len(Config.IDENTITY_COLS)

    def weighted_bce_loss(self, logits, targets, weights):
        """
        Computes Binary Cross Entropy with sample-specific weights.

        Args:
            logits: Model predictions (before sigmoid).
            targets: Binary targets (0 or 1).
            weights: Per-sample weights (higher for bias traps).
        """
        # Ensure inputs are flattened and float
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        weights = weights.view(-1).float()

        # Calculate raw per-sample loss
        loss = self.bce_none(logits, targets)

        # Apply bias-centric weights
        weighted_loss = loss * weights

        # Return mean loss
        return weighted_loss.mean()

    def pairwise_ranking_loss(self, logits, targets):
        """
        Computes Margin Ranking Loss for Toxic vs Non-Toxic pairs.

        This effectively mines 'hard' pairs (e.g., Toxic vs Non-Toxic w/ Identity)
        because 'easy' pairs (where Toxic score >> Non-Toxic score) will result
        in 0 loss due to the ReLU.
        """
        scores = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        # Identify positive (Toxic) and negative (Non-Toxic) examples
        # Threshold is 0.5 as per task definition
        pos_mask = targets >= 0.5
        neg_mask = targets < 0.5

        # If batch doesn't contain both classes, ranking loss is undefined (return 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # Create all pairs using broadcasting
        # pos_scores: (P, 1)
        # neg_scores: (1, N)
        # diff: (P, N) matrix of (pos_score_i - neg_score_j)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        # Optimization Goal: pos_score > neg_score + margin
        # Violation (Loss): margin - (pos - neg) > 0
        loss_matrix = F.relu(self.margin - diff)

        # Return mean loss over all valid pairs
        return loss_matrix.mean()

    def forward(self, outputs, batch):
        """
        Computes the total hybrid loss.

        Args:
            outputs (dict): Dictionary containing model logits ('toxicity_logits', etc.).
            batch (dict): Dictionary containing targets and weights.

        Returns:
            dict: Dictionary containing 'loss' (total) and individual components.
        """
        # Unpack Model Outputs
        tox_logits = outputs["toxicity_logits"]
        id_logits = outputs["identity_logits"]
        att_logits = outputs["attack_logits"]

        # Unpack Batch Data
        targets = batch["target"]
        aux_targets = batch["aux_targets"]
        weights = batch["weight"]

        # 1. Primary Loss: Weighted BCE on Toxicity
        loss_main = self.weighted_bce_loss(tox_logits, targets, weights)

        # 2. Auxiliary Loss: BCE on Identity and Attack Heads
        # The aux_targets tensor contains concatenated [Identities, Attack_Types]
        # We split them based on the number of identity columns
        id_targets = aux_targets[:, : self.num_identities]
        att_targets = aux_targets[:, self.num_identities :]

        loss_id = self.bce_mean(id_logits, id_targets)
        loss_att = self.bce_mean(att_logits, att_targets)

        # Combine auxiliary losses (simple average)
        loss_aux = (loss_id + loss_att) / 2.0

        # 3. Ranking Loss: Pairwise Margin
        loss_rank = self.pairwise_ranking_loss(tox_logits, targets)

        # Combine all losses
        # L_total = L_main + alpha * L_rank + beta * L_aux
        total_loss = loss_main + (self.alpha * loss_rank) + (self.beta * loss_aux)

        return {
            "loss": total_loss,
            "loss_main": loss_main,
            "loss_rank": loss_rank,
            "loss_aux": loss_aux,
        }
