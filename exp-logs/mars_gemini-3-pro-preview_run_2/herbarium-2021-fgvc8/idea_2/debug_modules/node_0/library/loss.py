import torch
import torch.nn as nn
from library.config import Config


class HierarchicalLoss(nn.Module):
    """
    Hierarchical Loss function that computes a weighted sum of losses for
    Species (ArcFace), Family, and Order heads.

    Handles CutMix regularization by computing the loss as a linear combination
    of losses for the two target labels based on the mixing coefficient lambda.
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()

        # Load loss weights from configuration
        self.w_species = Config.LOSS_WEIGHT_SPECIES
        self.w_family = Config.LOSS_WEIGHT_FAMILY
        self.w_order = Config.LOSS_WEIGHT_ORDER

        # Initialize CrossEntropyLoss with Label Smoothing
        # Note: ArcFace logits are compatible with CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def forward(self, outputs, targets):
        """
        Computes the total hierarchical loss.

        Args:
            outputs (tuple): A tuple containing logits from the model:
                             (species_logits, family_logits, order_logits)
            targets (dict): A dictionary containing targets and CutMix lambda:
                            {
                                'species': (target_a, target_b),
                                'family': (target_a, target_b),
                                'order': (target_a, target_b),
                                'lam': float
                            }
                            target_a and target_b are Tensors of shape [Batch_Size].
                            lam is a scalar (float).

        Returns:
            torch.Tensor: The computed weighted total loss (scalar).
        """
        # Unpack model outputs
        species_logits, family_logits, order_logits = outputs

        # Unpack targets
        # The CutMixCollator returns tuples of (target_a, target_b) for each label type
        species_targets = targets["species"]
        family_targets = targets["family"]
        order_targets = targets["order"]
        lam = targets["lam"]

        # 1. Species Loss
        # For ArcFace, the model output (species_logits) already includes the margin
        # penalty for the primary target (target_a) if passed correctly in the forward pass.
        loss_species = self._mix_loss(species_logits, species_targets, lam)

        # 2. Family Loss
        loss_family = self._mix_loss(family_logits, family_targets, lam)

        # 3. Order Loss
        loss_order = self._mix_loss(order_logits, order_targets, lam)

        # 4. Weighted Sum
        total_loss = (
            (self.w_species * loss_species)
            + (self.w_family * loss_family)
            + (self.w_order * loss_order)
        )

        return total_loss

    def _mix_loss(self, logits, targets_tuple, lam):
        """
        Helper to compute the mixed loss: lam * Loss(a) + (1 - lam) * Loss(b)

        Args:
            logits (torch.Tensor): Predictions [B, Num_Classes]
            targets_tuple (tuple): (target_a, target_b)
            lam (float): Mixing coefficient

        Returns:
            torch.Tensor: Scalar loss
        """
        target_a, target_b = targets_tuple

        # Calculate loss for both targets
        loss_a = self.criterion(logits, target_a)
        loss_b = self.criterion(logits, target_b)

        # Combine
        return lam * loss_a + (1 - lam) * loss_b
