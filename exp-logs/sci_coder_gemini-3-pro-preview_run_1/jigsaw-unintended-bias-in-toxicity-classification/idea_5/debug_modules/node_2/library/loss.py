import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TriangulationLoss(nn.Module):
    """
    Composite loss function for Semantic Triangulation.

    Computes a weighted sum of:
    1. Primary Toxicity Loss (Weighted BCE based on Bias Trap weights)
    2. Identity Loss (BCE, unweighted by sample weights)
    3. Identity Attack Loss (BCE, unweighted by sample weights)

    Formula:
    L_total = L_primary + (lambda_identity * L_identity) + (lambda_attack * L_attack)
    """

    def __init__(self):
        super(TriangulationLoss, self).__init__()
        self.lambda_identity = Config.LAMBDA_IDENTITY
        self.lambda_attack = Config.LAMBDA_ATTACK

    def forward(self, outputs, batch):
        """
        Calculates the composite loss.

        Args:
            outputs: Dictionary containing model logits:
                     - 'primary': (Batch, 1)
                     - 'identity': (Batch, Num_Identities)
                     - 'attack': (Batch, 1)
            batch: Dictionary containing targets and weights:
                   - 'target': (Batch,)
                   - 'identity_targets': (Batch, Num_Identities)
                   - 'attack_target': (Batch,)
                   - 'sample_weight': (Batch,)

        Returns:
            loss: Scalar tensor representing the total weighted loss.
            loss_dict: Dictionary containing individual loss components for logging.
        """
        # --- 1. Primary Toxicity Loss ---
        # Apply bias-centric sample weights here.
        # We use functional BCEWithLogits to pass per-sample weights manually.

        primary_logits = outputs["primary"]
        primary_targets = batch["target"].view(-1, 1)
        sample_weights = batch["sample_weight"].view(-1, 1)

        # binary_cross_entropy_with_logits handles the sigmoid internally
        loss_primary = F.binary_cross_entropy_with_logits(
            primary_logits, primary_targets, weight=sample_weights, reduction="mean"
        )

        # --- 2. Auxiliary Identity Loss ---
        # Multi-label classification for identity attributes.
        # No sample weights applied (preserve natural distribution).

        identity_logits = outputs["identity"]
        identity_targets = batch["identity_targets"]

        loss_identity = F.binary_cross_entropy_with_logits(
            identity_logits, identity_targets, reduction="mean"
        )

        # --- 3. Auxiliary Identity Attack Loss ---
        # Binary classification for 'identity_attack' subtype.
        # No sample weights applied.

        attack_logits = outputs["attack"]
        attack_targets = batch["attack_target"].view(-1, 1)

        loss_attack = F.binary_cross_entropy_with_logits(
            attack_logits, attack_targets, reduction="mean"
        )

        # --- 4. Total Loss Aggregation ---
        total_loss = (
            loss_primary
            + (self.lambda_identity * loss_identity)
            + (self.lambda_attack * loss_attack)
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_primary": loss_primary.item(),
            "loss_identity": loss_identity.item(),
            "loss_attack": loss_attack.item(),
        }

        return total_loss, loss_dict
